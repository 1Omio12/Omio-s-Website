import os
import uuid
import json
import base64
import shutil
import sqlite3
import logging
import h5py
import xml.etree.ElementTree as ET
from io import BytesIO

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from celery import Celery

# Optional heavy deps (OpenMC/Matplotlib) loaded lazily when needed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ----------------------------
# App and Celery configuration
# ----------------------------

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.before_request
def before_request():
    # hook for headers or auth if needed
    pass


@app.after_request
def after_request(response):
    response.headers['ngrok-skip-browser-warning'] = 'any'
    return response


REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
app.config['CELERY_BROKER_URL'] = REDIS_URL
app.config['CELERY_RESULT_BACKEND'] = REDIS_URL

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


# ----------------------------
# OpenMC configuration
# ----------------------------

# Update to your server paths
CROSS_SECTIONS_LIBRARIES = {
    'jeff-3.3': os.environ.get('OPENMC_XS_JEFF33', '/home/omio/Code/OpenMc/jeff-3.3-hdf5/cross_sections.xml'),
    'endfb-viii.0': os.environ.get('OPENMC_XS_ENDFBVIII0', '/home/omio/Code/OpenMc/endfb-viii.0-hdf5/cross_sections.xml'),
    'endfb-vii.1': os.environ.get('OPENMC_XS_ENDFBVII1', '/home/omio/Code/OpenMc/endfb-vii.1-hdf5/cross_sections.xml')
}


# ----------------------------
# Database helpers
# ----------------------------

def init_db():
    conn = sqlite3.connect('jobs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS jobs
                 (job_id TEXT PRIMARY KEY,
                  email TEXT,
                  status TEXT,
                  file_paths TEXT,
                  job_dir TEXT,
                  cross_sections TEXT DEFAULT 'jeff-3.3',
                  job_type TEXT DEFAULT 'xml',
                  code TEXT DEFAULT NULL,
                  plot_params TEXT DEFAULT NULL)''')

    # Migration for legacy tables: add columns if missing
    c.execute('PRAGMA table_info(jobs)')
    existing = {row[1] for row in c.fetchall()}
    if 'job_type' not in existing:
        try:
            c.execute("ALTER TABLE jobs ADD COLUMN job_type TEXT DEFAULT 'xml'")
        except Exception as e:
            logger.warning(f"job_type add failed: {e}")
    if 'code' not in existing:
        try:
            c.execute('ALTER TABLE jobs ADD COLUMN code TEXT DEFAULT NULL')
        except Exception as e:
            logger.warning(f"code add failed: {e}")
    if 'plot_params' not in existing:
        try:
            c.execute('ALTER TABLE jobs ADD COLUMN plot_params TEXT DEFAULT NULL')
        except Exception as e:
            logger.warning(f"plot_params add failed: {e}")

    conn.commit()
    conn.close()


# ----------------------------
# Utility helpers
# ----------------------------

def save_uploaded_file(file, job_dir, filename):
    try:
        secure_name = secure_filename(filename)
        file_path = os.path.join(job_dir, secure_name)
        os.makedirs(job_dir, exist_ok=True)
        file.save(file_path)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"Saved {secure_name} -> {file_path}")
            return file_path
        return None
    except Exception as e:
        logger.error(f"Save upload error {filename}: {e}")
        return None


def append_log(job_dir, text):
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, 'log.txt'), 'a', encoding='utf-8') as f:
            f.write(text)
            if not text.endswith('\n'):
                f.write('\n')
    except Exception as e:
        logger.warning(f"append_log failed: {e}")


# ----------------------------
# Geometry plotting helpers
# ----------------------------

def analyze_geometry_bounds(geometry_path):
    try:
        tree = ET.parse(geometry_path)
        root = tree.getroot()
        analysis = {
            'spheres': [], 'cylinders': [], 'planes': [], 'boxes': [],
            'max_radius': 0.0, 'max_coordinate': 0.0, 'geometry_type': 'unknown'
        }
        for sphere in root.findall('.//sphere'):
            r = sphere.get('r')
            if r:
                try:
                    rv = float(r)
                    analysis['spheres'].append(rv)
                    analysis['max_radius'] = max(analysis['max_radius'], rv)
                    if rv > 0:
                        analysis['geometry_type'] = 'spherical'
                except Exception:
                    pass
        for cyl in root.findall('.//cylinder'):
            r = cyl.get('r')
            if r:
                try:
                    rv = float(r)
                    analysis['cylinders'].append(rv)
                    analysis['max_radius'] = max(analysis['max_radius'], rv)
                    if rv > 0:
                        analysis['geometry_type'] = 'cylindrical'
                except Exception:
                    pass
        coords = []
        for plane in root.findall('.//plane'):
            for attr in ['x0', 'y0', 'z0']:
                v = plane.get(attr)
                if v:
                    try:
                        val = abs(float(v))
                        coords.append(val)
                        analysis['max_coordinate'] = max(analysis['max_coordinate'], val)
                    except Exception:
                        pass
        for box in root.findall('.//box'):
            for attr in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
                v = box.get(attr)
                if v:
                    try:
                        val = abs(float(v))
                        coords.append(val)
                        analysis['max_coordinate'] = max(analysis['max_coordinate'], val)
                        analysis['geometry_type'] = 'box'
                    except Exception:
                        pass
        analysis['planes'] = coords
        if analysis['max_radius'] > 0:
            dimension = analysis['max_radius'] * 2.4
            width = height = dimension
        elif analysis['max_coordinate'] > 0:
            dimension = analysis['max_coordinate'] * 2.4
            width = height = dimension
        else:
            width = height = 20.0
        width = max(1.0, min(width, 1000.0))
        height = max(1.0, min(height, 1000.0))
        origin = (0.0, 0.0, 0.0)
        return width, height, origin, analysis
    except Exception as e:
        logger.warning(f"analyze_geometry_bounds error: {e}")
        return 20.0, 20.0, (0.0, 0.0, 0.0), {'error': str(e)}


def calculate_optimal_resolution(width, height, geometry_type, num_materials):
    base_pixels = 800
    size_factor = max(width, height)
    if size_factor > 100:
        resolution_multiplier = 0.8
    elif size_factor > 10:
        resolution_multiplier = 1.0
    elif size_factor > 1:
        resolution_multiplier = 1.5
    else:
        resolution_multiplier = 2.0
    if geometry_type in ['cylindrical', 'spherical']:
        resolution_multiplier *= 1.2
    if num_materials > 5:
        resolution_multiplier *= 1.1
    pixels = int(base_pixels * resolution_multiplier)
    return max(400, min(pixels, 2400))


def create_adaptive_geometry_plots(materials, geometry_path, job_dir, plot_params=None):
    import openmc

    try:
        user_width = None
        user_height = None
        user_resolution = None
        user_basis = 'auto'
        user_origin = (0.0, 0.0, 0.0)
        if plot_params:
            if plot_params.get('width'):
                try:
                    user_width = float(plot_params['width'])
                except Exception:
                    pass
            if plot_params.get('height'):
                try:
                    user_height = float(plot_params['height'])
                except Exception:
                    pass
            if plot_params.get('resolution') and plot_params['resolution'] != 'auto':
                try:
                    user_resolution = int(plot_params['resolution'])
                except Exception:
                    pass
            user_basis = plot_params.get('basis', 'auto')
            try:
                origin_x = float(plot_params.get('origin_x', 0.0))
                origin_y = float(plot_params.get('origin_y', 0.0))
                origin_z = float(plot_params.get('origin_z', 0.0))
                user_origin = (origin_x, origin_y, origin_z)
            except Exception:
                user_origin = (0.0, 0.0, 0.0)

        if user_width is None or user_height is None:
            auto_width, auto_height, auto_origin, analysis = analyze_geometry_bounds(geometry_path)
            width = user_width if user_width is not None else auto_width
            height = user_height if user_height is not None else auto_height
            origin = user_origin if plot_params else auto_origin
        else:
            width = user_width
            height = user_height
            origin = user_origin
            analysis = {'geometry_type': 'user_defined'}

        if user_resolution is not None:
            main_resolution = user_resolution
        else:
            num_materials = len(materials)
            geometry_type = analysis.get('geometry_type', 'unknown')
            main_resolution = calculate_optimal_resolution(width, height, geometry_type, num_materials)
            side_resolution = int(main_resolution * 0.75)

        color_palette = [
            'red', 'blue', 'green', 'orange', 'purple', 'cyan',
            'magenta', 'yellow', 'brown', 'pink', 'gray', 'olive',
            'navy', 'lime', 'maroon', 'teal', 'silver', 'gold'
        ]

        plots_to_create = []

        def set_colors(plot_obj):
            plot_obj.colors = {}
            for i, material in enumerate(materials):
                try:
                    material_id = int(material.id)
                    if material_id > 0:
                        plot_obj.colors[material_id] = color_palette[i % len(color_palette)]
                except Exception:
                    continue

        if user_basis == 'auto':
            plot_xy = openmc.Plot()
            plot_xy.filename = 'geometry_plot'
            plot_xy.width = (width, height)
            plot_xy.pixels = (main_resolution, main_resolution)
            plot_xy.color_by = 'material'
            plot_xy.basis = 'xy'
            plot_xy.origin = origin
            set_colors(plot_xy)
            plots_to_create.append(plot_xy)

            geometry_type = analysis.get('geometry_type', 'unknown')
            num_materials = len(materials)
            if (analysis.get('max_coordinate', 0) > 0 or geometry_type in ['cylindrical', 'box'] or num_materials > 2):
                plot_xz = openmc.Plot()
                plot_xz.filename = 'geometry_plot_xz'
                plot_xz.width = (width, width)
                plot_xz.pixels = (side_resolution, side_resolution)
                plot_xz.color_by = 'material'
                plot_xz.basis = 'xz'
                plot_xz.origin = origin
                set_colors(plot_xz)
                plots_to_create.append(plot_xz)

                if num_materials > 3 or geometry_type == 'box':
                    plot_yz = openmc.Plot()
                    plot_yz.filename = 'geometry_plot_yz'
                    plot_yz.width = (height, width)
                    plot_yz.pixels = (side_resolution, side_resolution)
                    plot_yz.color_by = 'material'
                    plot_yz.basis = 'yz'
                    plot_yz.origin = origin
                    set_colors(plot_yz)
                    plots_to_create.append(plot_yz)
        else:
            plot = openmc.Plot()
            plot.filename = f'geometry_plot_{user_basis}'
            plot.width = (width, height)
            plot.pixels = (main_resolution, main_resolution)
            plot.color_by = 'material'
            plot.basis = user_basis
            plot.origin = origin
            set_colors(plot)
            plots_to_create.append(plot)

        if user_basis == 'auto' and max(width, height) < 5.0:
            plot_zoom = openmc.Plot()
            plot_zoom.filename = 'geometry_plot_zoom'
            plot_zoom.width = (width * 0.5, height * 0.5)
            plot_zoom.pixels = (int(main_resolution * 1.5), int(main_resolution * 1.5))
            plot_zoom.color_by = 'material'
            plot_zoom.basis = 'xy'
            plot_zoom.origin = origin
            set_colors(plot_zoom)
            plots_to_create.append(plot_zoom)

        import openmc
        plots = openmc.Plots(plots_to_create)
        plots.export_to_xml()
        try:
            openmc.plot_geometry()
        except Exception as e:
            logger.warning(f"plot_geometry failed: {e}")
            return False, []

        created_plots = []
        for pf in ['geometry_plot.png', 'geometry_plot_xz.png', 'geometry_plot_yz.png', 'geometry_plot_zoom.png']:
            if os.path.exists(pf):
                created_plots.append(pf)

        enhanced_plots = []
        for plot_file in created_plots:
            try:
                image = plt.imread(plot_file)
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(image)
                if '_xz' in plot_file:
                    xlabel, ylabel = 'X (cm)', 'Z (cm)'
                elif '_yz' in plot_file:
                    xlabel, ylabel = 'Y (cm)', 'Z (cm)'
                else:
                    xlabel, ylabel = 'X (cm)', 'Y (cm)'
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                handles = []
                for i, material in enumerate(materials):
                    try:
                        mid = int(material.id)
                        color = plots_to_create[0].colors.get(mid, 'black')
                        label = material.name if material.name else f'Material {mid}'
                        handles.append(mpatches.Patch(color=color, label=label))
                    except Exception:
                        continue
                if handles:
                    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                enhanced_file = plot_file.replace('.png', '_enhanced.png')
                plt.savefig(enhanced_file, bbox_inches='tight')
                plt.close()
                enhanced_plots.append(enhanced_file)
            except Exception as e:
                logger.warning(f"Enhance failed {plot_file}: {e}")
                enhanced_plots.append(plot_file)

        return (len(enhanced_plots) > 0), enhanced_plots
    except Exception as e:
        logger.error(f"create_adaptive_geometry_plots error: {e}")
        return False, []


# ----------------------------
# Routes: core pages
# ----------------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/editor')
def editor():
    return render_template('editor.html')


# ----------------------------
# XML submission flow (existing behavior)
# ----------------------------

@app.route('/submit', methods=['POST'])
def submit_job():
    try:
        import openmc  # ensure module is importable now

        job_id = str(uuid.uuid4())
        email = request.form.get('email', 'test@example.com')
        cross_sections = request.form.get('cross_sections', 'jeff-3.3')
        plot_params = {
            'width': request.form.get('plot_width'),
            'height': request.form.get('plot_height'),
            'resolution': request.form.get('plot_resolution', 'auto'),
            'basis': request.form.get('plot_basis', 'auto'),
            'origin_x': request.form.get('origin_x', '0.0'),
            'origin_y': request.form.get('origin_y', '0.0'),
            'origin_z': request.form.get('origin_z', '0.0')
        }
        plot_params = {k: v for k, v in plot_params.items() if v not in (None, '',)}

        if cross_sections not in CROSS_SECTIONS_LIBRARIES:
            return jsonify({'error': f'Invalid cross-section library: {cross_sections}'}), 400
        xs_path = CROSS_SECTIONS_LIBRARIES[cross_sections]
        if not os.path.exists(xs_path):
            return jsonify({'error': f'Cross-section library not found: {xs_path}'}), 400

        job_dir = os.path.abspath(os.path.join('jobs', job_id))
        os.makedirs(job_dir, exist_ok=True)

        required_files = ['geometry', 'materials', 'settings']
        optional_files = ['tallies', 'depletion']
        file_paths = {}

        for ftype in required_files:
            if ftype not in request.files:
                return jsonify({'error': f'Missing {ftype}.xml file'}), 400
            f = request.files[ftype]
            if f.filename == '':
                return jsonify({'error': f'No {ftype}.xml file selected'}), 400
            saved = save_uploaded_file(f, job_dir, f"{ftype}.xml")
            if not saved:
                return jsonify({'error': f'Failed to save {ftype}.xml file'}), 500
            file_paths[ftype] = saved

        for ftype in optional_files:
            if ftype in request.files and request.files[ftype].filename:
                f = request.files[ftype]
                fname = f"{ftype}.xml" if ftype == 'tallies' else f"{ftype}.ipynb"
                saved = save_uploaded_file(f, job_dir, fname)
                if saved:
                    file_paths[ftype] = saved

        # persist job
        data = {'file_paths': file_paths, 'plot_params': plot_params}
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('''INSERT INTO jobs (job_id, email, status, file_paths, job_dir, cross_sections, job_type)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (job_id, email, 'queued', json.dumps(data), job_dir, cross_sections, 'xml'))
        conn.commit()
        conn.close()

        run_simulation.delay(job_id)
        return jsonify({'job_id': job_id}), 202
    except Exception as e:
        logger.exception('submit_job failed')
        return jsonify({'error': str(e)}), 500


# ----------------------------
# Code-editor submission flow
# ----------------------------

@app.route('/code/submit', methods=['POST'])
def submit_code_job():
    try:
        code = request.form.get('code', '')
        email = request.form.get('email', 'test@example.com')
        cross_sections = request.form.get('cross_sections', 'jeff-3.3')
        if not code.strip():
            return jsonify({'error': 'Empty code body'}), 400
        if cross_sections not in CROSS_SECTIONS_LIBRARIES:
            return jsonify({'error': f'Invalid cross-section library: {cross_sections}'}), 400

        plot_params = {
            'width': request.form.get('plot_width'),
            'height': request.form.get('plot_height'),
            'resolution': request.form.get('plot_resolution', 'auto'),
            'basis': request.form.get('plot_basis', 'auto'),
            'origin_x': request.form.get('origin_x', '0.0'),
            'origin_y': request.form.get('origin_y', '0.0'),
            'origin_z': request.form.get('origin_z', '0.0')
        }
        plot_params = {k: v for k, v in plot_params.items() if v not in (None, '',)}

        job_id = str(uuid.uuid4())
        job_dir = os.path.abspath(os.path.join('jobs', job_id))
        os.makedirs(job_dir, exist_ok=True)

        # Save code and seed helper runner
        code_path = os.path.join(job_dir, 'main.py')
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code)

        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('''INSERT INTO jobs (job_id, email, status, file_paths, job_dir, cross_sections, job_type, code, plot_params)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                  (job_id, email, 'queued', json.dumps({}), job_dir, cross_sections, 'code', code, json.dumps(plot_params)))
        conn.commit()
        conn.close()

        run_code_job.delay(job_id)
        return jsonify({'job_id': job_id}), 202
    except Exception as e:
        logger.exception('submit_code_job failed')
        return jsonify({'error': str(e)}), 500


@app.route('/logs/<job_id>')
def get_logs(job_id):
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT job_dir FROM jobs WHERE job_id = ?', (job_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Job not found'}), 404
        job_dir = row[0]
        path = os.path.join(job_dir, 'log.txt')
        if not os.path.exists(path):
            return jsonify({'log': ''})
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            return jsonify({'log': f.read()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/download/<job_id>/<path:filename>')
def download_artifact(job_id, filename):
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT job_dir FROM jobs WHERE job_id = ?', (job_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Job not found'}), 404
        job_dir = row[0]
        return send_from_directory(job_dir, filename, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------
# Core simulation task
# ----------------------------

@celery.task
def run_simulation(job_id):
    import openmc

    conn = None
    original_dir = os.getcwd()
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT file_paths, job_dir, cross_sections, plot_params FROM jobs WHERE job_id = ?', (job_id,))
        row = c.fetchone()
        if not row:
            raise ValueError(f"Job {job_id} not found")
        file_paths_json, job_dir, cross_sections, plot_params_json = row
        try:
            data = json.loads(file_paths_json) if file_paths_json else {}
            if isinstance(data, dict) and 'file_paths' in data:
                file_paths = data['file_paths']
                plot_params = data.get('plot_params', {})
            else:
                file_paths = data or {}
                plot_params = json.loads(plot_params_json) if plot_params_json else {}
        except Exception:
            file_paths = {}
            plot_params = {}

        if cross_sections not in CROSS_SECTIONS_LIBRARIES:
            raise ValueError('Invalid cross-sections selection')
        xs_path = CROSS_SECTIONS_LIBRARIES[cross_sections]
        if not os.path.exists(xs_path):
            raise ValueError(f'Cross-section library not found: {xs_path}')

        for key in ['geometry', 'materials', 'settings']:
            if key not in file_paths:
                raise ValueError(f'Missing file path for {key}')
            if not os.path.exists(file_paths[key]):
                raise FileNotFoundError(f"{key}.xml not found at {file_paths[key]}")
            if os.path.getsize(file_paths[key]) == 0:
                raise ValueError(f"{key}.xml is empty: {file_paths[key]}")

        os.chdir(job_dir)
        os.environ['OPENMC_CROSS_SECTIONS'] = xs_path
        c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', ('running', job_id))
        conn.commit()

        try:
            materials = openmc.Materials.from_xml(file_paths['materials'])
            geometry = openmc.Geometry.from_xml(file_paths['geometry'])
            settings = openmc.Settings.from_xml(file_paths['settings'])
            model = openmc.Model(geometry=geometry, materials=materials, settings=settings)
            if 'tallies' in file_paths and os.path.exists(file_paths['tallies']):
                model.tallies = openmc.Tallies.from_xml(file_paths['tallies'])
        except Exception as e:
            append_log(job_dir, f"Model load error: {e}")
            raise

        try:
            append_log(job_dir, 'Starting OpenMC simulation...')
            model.run()
            append_log(job_dir, 'OpenMC simulation completed successfully')
        except Exception as e:
            append_log(job_dir, f"OpenMC simulation failed: {e}")
            raise

        try:
            success, geometry_plots = create_adaptive_geometry_plots(materials, file_paths['geometry'], job_dir, plot_params)
            if success:
                append_log(job_dir, f"Generated {len(geometry_plots)} geometry plots")
        except Exception as e:
            append_log(job_dir, f"Plot generation error: {e}")

        c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', ('completed', job_id))
        conn.commit()
    except Exception as e:
        if conn:
            c = conn.cursor()
            c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', (f'failed: {e}', job_id))
            conn.commit()
        raise
    finally:
        os.chdir(original_dir)
        if conn:
            conn.close()


@celery.task
def run_code_job(job_id):
    """Runs user Python code to generate OpenMC XMLs, then enqueue simulation."""
    conn = None
    original_dir = os.getcwd()
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT job_dir, cross_sections, code, plot_params FROM jobs WHERE job_id = ?', (job_id,))
        row = c.fetchone()
        if not row:
            raise ValueError('Job not found')
        job_dir, cross_sections, code, plot_params_json = row
        plot_params = json.loads(plot_params_json) if plot_params_json else {}

        xs_path = CROSS_SECTIONS_LIBRARIES.get(cross_sections)
        if not xs_path or not os.path.exists(xs_path):
            raise ValueError('Cross sections not available')

        os.chdir(job_dir)
        c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', ('running', job_id))
        conn.commit()

        append_log(job_dir, 'Running user code to generate OpenMC XML...')
        env = os.environ.copy()
        env['OPENMC_CROSS_SECTIONS'] = xs_path

        # Use python -I for isolated mode; enforce timeout
        import subprocess
        try:
            proc = subprocess.Popen(
                ['python', '-I', 'main.py'],
                cwd=job_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
                bufsize=1,
            )
            log_path = os.path.join(job_dir, 'log.txt')
            with open(log_path, 'a', encoding='utf-8') as lf:
                for line in proc.stdout:
                    lf.write(line)
            ret = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            append_log(job_dir, 'Error: Code execution timed out (600s)')
            try:
                proc.kill()
            except Exception:
                pass
            ret = -1
        except Exception as e:
            append_log(job_dir, f'Execution failed: {e}')
            ret = -1

        if ret != 0:
            append_log(job_dir, f'Code execution returned non-zero exit status: {ret}')
            c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', ('failed: code execution error', job_id))
            conn.commit()
            return

        # Detect generated XMLs
        files = {
            'materials': os.path.join(job_dir, 'materials.xml'),
            'geometry': os.path.join(job_dir, 'geometry.xml'),
            'settings': os.path.join(job_dir, 'settings.xml'),
        }
        if os.path.exists(os.path.join(job_dir, 'tallies.xml')):
            files['tallies'] = os.path.join(job_dir, 'tallies.xml')

        missing = [k for k, v in files.items() if not os.path.exists(v)]
        if missing:
            append_log(job_dir, f"Missing required XML files: {', '.join(missing)}")
            c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', ('failed: XML generation missing files', job_id))
            conn.commit()
            return

        # Update DB with file paths then enqueue simulation
        job_data = {'file_paths': files, 'plot_params': plot_params}
        c.execute('UPDATE jobs SET file_paths = ?, status = ? WHERE job_id = ?', (json.dumps(job_data), 'queued', job_id))
        conn.commit()
        run_simulation.delay(job_id)
    except Exception as e:
        if conn:
            c = conn.cursor()
            c.execute('UPDATE jobs SET status = ? WHERE job_id = ?', (f'failed: {e}', job_id))
            conn.commit()
        raise
    finally:
        os.chdir(original_dir)
        if conn:
            conn.close()


# ----------------------------
# Results and status
# ----------------------------

@app.route('/status/<job_id>')
def get_status(job_id):
    conn = sqlite3.connect('jobs.db')
    c = conn.cursor()
    c.execute('SELECT status FROM jobs WHERE job_id = ?', (job_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify({'status': row[0]})


@app.route('/results/<job_id>')
def show_results(job_id):
    import openmc

    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT status, job_dir, cross_sections, file_paths FROM jobs WHERE job_id = ?', (job_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return render_template('results.html', job_id=job_id, status='Job not found', results=None, plot_data=None, geometry_views={}, cross_sections_used=None)

        status, job_dir, cross_sections_used, file_paths_json = row
        try:
            data = json.loads(file_paths_json) if file_paths_json else {}
        except Exception:
            data = {}

        file_paths = data.get('file_paths', data) if isinstance(data, dict) else {}
        results = []
        plot_data = None
        geometry_views = {}

        if status == 'completed' and os.path.exists(job_dir):
            try:
                sp_files = [f for f in os.listdir(job_dir) if f.startswith('statepoint') and f.endswith('.h5')]
                if sp_files:
                    sp_file = os.path.join(job_dir, sorted(sp_files)[-1])
                    try:
                        sp = openmc.StatePoint(sp_file)
                        if hasattr(sp, 'keff'):
                            results.append({'metric': 'k-effective', 'value': f"{sp.keff.nominal_value:.6f}", 'uncertainty': f"± {sp.keff.std_dev:.6f}"})
                        if hasattr(sp, 'n_batches'):
                            results.append({'metric': 'Number of batches', 'value': f"{sp.n_batches}", 'uncertainty': 'N/A'})
                        if hasattr(sp, 'n_inactive'):
                            results.append({'metric': 'Inactive batches', 'value': f"{sp.n_inactive}", 'uncertainty': 'N/A'})
                        if hasattr(sp, 'n_particles'):
                            results.append({'metric': 'Particles per batch', 'value': f"{sp.n_particles}", 'uncertainty': 'N/A'})
                        if sp.tallies:
                            means = []
                            labels = []
                            for tid, tally in sp.tallies.items():
                                try:
                                    m = tally.mean.flatten()[0]
                                    s = tally.std_dev.flatten()[0] if hasattr(tally, 'std_dev') else 0.0
                                    results.append({'metric': f'Tally {tid}', 'value': f"{m:.6e}", 'uncertainty': f"± {s:.6e}"})
                                    means.append(m)
                                    labels.append(f'Tally {tid}')
                                except Exception:
                                    continue
                            if means:
                                plt.figure(figsize=(12, 8))
                                bars = plt.bar(labels, means, color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'])
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                buf = BytesIO()
                                plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
                                buf.seek(0)
                                plot_data = base64.b64encode(buf.getvalue()).decode()
                                plt.close()
                    except Exception as e:
                        results.append({'metric': 'Statepoint', 'value': os.path.basename(sp_file), 'uncertainty': f'Parse error: {e}'})
                else:
                    results.append({'metric': 'File Status', 'value': 'No statepoint files found', 'uncertainty': 'Simulation may have failed or not produced output'})
            except Exception as e:
                results.append({'metric': 'Parse Error', 'value': str(e), 'uncertainty': 'Contact support'})

        # Geometry views
        if job_dir and os.path.exists(job_dir):
            original_cwd = os.getcwd()
            try:
                os.chdir(job_dir)
                patterns = {
                    'xy': ['geometry_plot_enhanced.png', 'geometry_plot.png'],
                    'xz': ['geometry_plot_xz_enhanced.png', 'geometry_plot_xz.png'],
                    'yz': ['geometry_plot_yz_enhanced.png', 'geometry_plot_yz.png'],
                    'xy_zoom': ['geometry_plot_zoom_enhanced.png', 'geometry_plot_zoom.png'],
                    'simple': ['geometry_plot_simple.png', 'geometry_plot_minimal.png']
                }
                for key, files in patterns.items():
                    for f in files:
                        if os.path.exists(f):
                            with open(f, 'rb') as imgf:
                                geometry_views[key] = base64.b64encode(imgf.read()).decode('utf-8')
                            break
            finally:
                os.chdir(original_cwd)

        return render_template('results.html', job_id=job_id, status=status, results=results,
                               plot_data=plot_data, geometry_views=geometry_views,
                               cross_sections_used=cross_sections_used, file_paths=file_paths, job_dir=job_dir)
    except Exception as e:
        logger.error(f"show_results error: {e}")
        return render_template('results.html', job_id=job_id, status=f'Error: {e}', results=[{'metric': 'System Error', 'value': f'{e}', 'uncertainty': 'Contact support'}], plot_data=None, geometry_views={}, cross_sections_used=None, file_paths={}, job_dir=None)


# ----------------------------
# Manual/preview plots
# ----------------------------

@app.route('/generate-manual-plot/<job_id>')
def generate_manual_plot(job_id):
    import openmc

    try:
        width = float(request.args.get('width', 20.0))
        height = float(request.args.get('height', 20.0))
        basis = request.args.get('basis', 'xy')
        resolution = int(request.args.get('resolution', 800))
        origin_x = float(request.args.get('origin_x', 0.0))
        origin_y = float(request.args.get('origin_y', 0.0))
        origin_z = float(request.args.get('origin_z', 0.0))

        width = max(0.1, min(width, 1000.0))
        height = max(0.1, min(height, 1000.0))
        resolution = max(100, min(resolution, 3000))
        basis = basis if basis in ['xy', 'xz', 'yz'] else 'xy'

        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT file_paths, job_dir FROM jobs WHERE job_id = ?', (job_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Job not found'}), 404
        file_paths_json, job_dir = row
        try:
            data = json.loads(file_paths_json) if file_paths_json else {}
            file_paths = data.get('file_paths', data)
        except Exception:
            return jsonify({'error': 'Invalid file paths data'}), 400

        if not os.path.exists(job_dir):
            return jsonify({'error': 'Job directory not found'}), 404

        original_dir = os.getcwd()
        try:
            os.chdir(job_dir)
            geometry_path = file_paths.get('geometry')
            materials_path = file_paths.get('materials')
            if not geometry_path or not materials_path:
                return jsonify({'error': 'Required file paths not found'}), 400
            if not os.path.exists(geometry_path) or not os.path.exists(materials_path):
                return jsonify({'error': 'Required geometry or materials files not found'}), 400

            materials = openmc.Materials.from_xml(materials_path)
            _ = openmc.Geometry.from_xml(geometry_path)

            plot_filename = f'manual_plot_{basis}_{width}x{height}_{resolution}'
            plot = openmc.Plot()
            plot.filename = plot_filename
            plot.width = (width, height)
            plot.pixels = (resolution, resolution)
            plot.color_by = 'material'
            plot.basis = basis
            plot.origin = (origin_x, origin_y, origin_z)
            palette = ['red','blue','green','orange','purple','cyan','magenta','yellow','brown','pink','gray','olive','navy','lime','maroon','teal','silver','gold']
            plot.colors = {}
            for i, m in enumerate(materials):
                try:
                    mid = int(m.id)
                    if mid > 0:
                        plot.colors[mid] = palette[i % len(palette)]
                except Exception:
                    continue
            plots = openmc.Plots([plot])
            plots.export_to_xml()
            openmc.plot_geometry()
            pf = f'{plot_filename}.png'
            if os.path.exists(pf):
                with open(pf, 'rb') as f:
                    return jsonify({'success': True, 'plot_data': base64.b64encode(f.read()).decode(), 'parameters': {'width': width, 'height': height, 'basis': basis, 'resolution': resolution, 'origin': [origin_x, origin_y, origin_z]}})
            return jsonify({'error': 'Failed to generate manual plot'}), 500
        finally:
            os.chdir(original_dir)
    except Exception as e:
        logger.error(f"generate_manual_plot error: {e}")
        return jsonify({'error': f'Failed to generate manual plot: {e}'}), 500


@app.route('/geometry-preview/<job_id>')
def geometry_preview(job_id):
    import openmc

    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT file_paths, job_dir FROM jobs WHERE job_id = ?', (job_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return jsonify({'error': 'Job not found'}), 404
        file_paths_json, job_dir = row
        try:
            data = json.loads(file_paths_json) if file_paths_json else {}
            file_paths = data.get('file_paths', data)
        except Exception:
            return jsonify({'error': 'Invalid file paths data'}), 400

        if not os.path.exists(job_dir):
            return jsonify({'error': 'Job directory not found'}), 404

        original_dir = os.getcwd()
        try:
            os.chdir(job_dir)
            geometry_path = file_paths.get('geometry')
            materials_path = file_paths.get('materials')
            if not geometry_path or not materials_path:
                return jsonify({'error': 'Required file paths not found'}), 400
            if not os.path.exists(geometry_path) or not os.path.exists(materials_path):
                return jsonify({'error': 'Required geometry or materials files not found'}), 400
            materials = openmc.Materials.from_xml(materials_path)
            _ = openmc.Geometry.from_xml(geometry_path)
            width, height, origin, analysis = analyze_geometry_bounds(geometry_path)
            num_materials = len(materials)
            geometry_type = analysis.get('geometry_type', 'unknown')
            resolution = calculate_optimal_resolution(width, height, geometry_type, num_materials)
            plot = openmc.Plot()
            plot.filename = 'geometry_preview'
            plot.width = (width, height)
            plot.pixels = (min(resolution, 800), min(resolution, 800))
            plot.color_by = 'material'
            plot.basis = 'xy'
            plot.origin = origin
            palette = ['red','blue','green','orange','purple','cyan','magenta','yellow','brown','pink','gray','olive','navy','lime','maroon','teal','silver','gold']
            plot.colors = {}
            for i, m in enumerate(materials):
                try:
                    mid = int(m.id)
                    if mid > 0:
                        plot.colors[mid] = palette[i % len(palette)]
                except Exception:
                    continue
            plots = openmc.Plots([plot])
            plots.export_to_xml()
            openmc.plot_geometry()
            preview_path = 'geometry_preview.png'
            if os.path.exists(preview_path):
                with open(preview_path, 'rb') as f:
                    preview_data = base64.b64encode(f.read()).decode()
                return jsonify({'success': True, 'preview': preview_data, 'message': 'Adaptive geometry preview generated successfully', 'analysis': {'geometry_type': geometry_type, 'dimensions': f'{width:.1f} x {height:.1f}', 'materials': num_materials, 'resolution': f"{plot.pixels[0]}x{plot.pixels[1]}"}})
            return jsonify({'error': 'Failed to generate geometry preview'}), 500
        finally:
            os.chdir(original_dir)
    except Exception as e:
        logger.error(f"geometry_preview error: {e}")
        return jsonify({'error': f'Failed to generate preview: {e}'}), 500


# ----------------------------
# Debug
# ----------------------------

@app.route('/debug/<job_id>')
def debug_job(job_id):
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute('SELECT * FROM jobs WHERE job_id = ?', (job_id,))
        result = c.fetchone()
        if not result:
            conn.close()
            return jsonify({'error': 'Job not found'}), 404
        c.execute('PRAGMA table_info(jobs)')
        columns = [r[1] for r in c.fetchall()]
        job_info = dict(zip(columns, result))
        conn.close()

        job_dir = job_info.get('job_dir', os.path.join('jobs', job_id))
        fs_info = {'job_dir_exists': os.path.exists(job_dir), 'job_dir_path': job_dir, 'files_in_dir': []}
        if os.path.exists(job_dir):
            try:
                for name in os.listdir(job_dir):
                    p = os.path.join(job_dir, name)
                    fs_info['files_in_dir'].append({'name': name, 'size': os.path.getsize(p) if os.path.exists(p) else 0, 'exists': os.path.exists(p)})
            except Exception as e:
                fs_info['error'] = str(e)

        file_paths_info = {}
        if job_info.get('file_paths'):
            try:
                data = json.loads(job_info['file_paths'])
                fp = data.get('file_paths', data)
                for k, p in fp.items():
                    file_paths_info[k] = {'path': p, 'exists': os.path.exists(p), 'size': os.path.getsize(p) if os.path.exists(p) else 0}
            except Exception as e:
                file_paths_info['error'] = str(e)

        return jsonify({'job_info': job_info, 'file_system': fs_info, 'file_paths': file_paths_info})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------
# Optional AI suggestion endpoint (Ollama/HuggingFace) - FREE-friendly
# ----------------------------

@app.route('/ai/suggest', methods=['POST'])
def ai_suggest():
    try:
        payload = request.get_json(force=True)
        prompt = payload.get('prompt', '')
        code = payload.get('code', '')
        model_hint = payload.get('model', os.environ.get('AI_MODEL', 'llama3.1'))
        provider = os.environ.get('AI_PROVIDER', 'disabled')  # 'ollama' or 'huggingface' or 'disabled'
        max_tokens = int(os.environ.get('AI_MAX_TOKENS', '256'))

        if provider == 'ollama':
            import requests
            data = {
                'model': model_hint,
                'prompt': f"You are an OpenMC assistant. Given the following code, suggest improvements or next steps.\n\nCODE:\n{code}\n\nQUESTION:\n{prompt}",
                'stream': False,
                'options': {'num_predict': max_tokens}
            }
            r = requests.post('http://localhost:11434/api/generate', json=data, timeout=60)
            r.raise_for_status()
            out = r.json().get('response', '')
            return jsonify({'suggestion': out})
        elif provider == 'huggingface':
            import requests
            hf_key = os.environ.get('HF_API_KEY')
            if not hf_key:
                return jsonify({'error': 'HF_API_KEY not set'}), 400
            headers = {'Authorization': f'Bearer {hf_key}'}
            model = model_hint or 'mistralai/Mistral-7B-Instruct-v0.3'
            r = requests.post(
                f'https://api-inference.huggingface.co/models/{model}',
                headers=headers,
                json={'inputs': f"SYSTEM: You are an OpenMC assistant helping write nuclear simulations.\nUSER: {prompt}\nCODE:\n{code}"},
                timeout=60,
            )
            r.raise_for_status()
            data = r.json()
            if isinstance(data, list) and data:
                text = data[0].get('generated_text', '') or data[0].get('summary_text', '')
            else:
                text = json.dumps(data)
            return jsonify({'suggestion': text})
        else:
            # Disabled provider: return a static tip
            tip = 'Tip: Build your model with openmc.Model, call export_to_xml(), then run. Example: define materials, geometry, settings, and tallies.'
            return jsonify({'suggestion': tip})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ----------------------------
# Entrypoint
# ----------------------------

if __name__ == '__main__':
    missing = []
    for name, path in CROSS_SECTIONS_LIBRARIES.items():
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")
    if missing:
        print('WARNING: Missing cross-sections:')
        for m in missing:
            print(f'  - {m}')
        print('Update CROSS_SECTIONS_LIBRARIES or set env vars OPENMC_XS_*')
    os.makedirs('jobs', exist_ok=True)
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)

