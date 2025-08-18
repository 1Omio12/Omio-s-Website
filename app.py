import os
import uuid
import json
import base64
import sqlite3
import logging
import subprocess
from io import BytesIO

from flask import Flask, request, jsonify, render_template
from celery import Celery
from werkzeug.utils import secure_filename

# Heavy deps are imported lazily in functions where needed

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------
# Flask and Celery setup
# ----------------------------------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')


@app.before_request
def before_request():
    # Reserved for headers or auth; keep minimal to avoid side-effects
    pass


@app.after_request
def after_request(response):
    # Bypass ngrok warning (front-end also sets this header)
    response.headers['ngrok-skip-browser-warning'] = 'any'
    return response


app.config['CELERY_BROKER_URL'] = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

# ----------------------------------------------------------------------------
# Cross sections config
# ----------------------------------------------------------------------------
default_cross_sections = {
    'jeff-3.3': '/home/omio/Code/OpenMc/jeff-3.3-hdf5/cross_sections.xml',
    'endfb-viii.0': '/home/omio/Code/OpenMc/endfb-viii.0-hdf5/cross_sections.xml',
    'endfb-vii.1': '/home/omio/Code/OpenMc/endfb-vii.1-hdf5/cross_sections.xml',
}

# Allow override via env var CROSS_SECTIONS_LIBRARIES_JSON (a JSON string)
if os.getenv('CROSS_SECTIONS_LIBRARIES_JSON'):
    try:
        CROSS_SECTIONS_LIBRARIES = json.loads(os.getenv('CROSS_SECTIONS_LIBRARIES_JSON'))
    except Exception as _e:
        logger.warning('Invalid CROSS_SECTIONS_LIBRARIES_JSON; using defaults')
        CROSS_SECTIONS_LIBRARIES = default_cross_sections
else:
    CROSS_SECTIONS_LIBRARIES = default_cross_sections


# ----------------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------------
DB_PATH = '/workspace/jobs.db'


def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS jobs
                 (job_id TEXT PRIMARY KEY,
                  email TEXT,
                  status TEXT,
                  file_paths TEXT,
                  job_dir TEXT,
                  cross_sections TEXT DEFAULT 'jeff-3.3',
                  mode TEXT DEFAULT 'xml' -- 'xml' uploads or 'code' editor
                 )''')
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def save_uploaded_file(file_storage, job_dir, filename):
    try:
        os.makedirs(job_dir, exist_ok=True)
        safe = secure_filename(filename)
        fpath = os.path.join(job_dir, safe)
        file_storage.save(fpath)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
            return fpath
        return None
    except Exception as e:
        logger.exception(f"Saving file failed: {e}")
        return None


# ----------------------------------------------------------------------------
# Lightweight imports of heavy libs
# ----------------------------------------------------------------------------
def _import_openmc_stack():
    import openmc
    import h5py
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import xml.etree.ElementTree as ET
    return openmc, h5py, matplotlib, plt, mpatches, ET


# ----------------------------------------------------------------------------
# Routes - UI pages
# ----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/editor')
def editor():
    return render_template('editor.html')


# ----------------------------------------------------------------------------
# Submit XML job (existing flow)
# ----------------------------------------------------------------------------
@app.route('/submit', methods=['POST'])
def submit_job():
    try:
        openmc, h5py, matplotlib, plt, mpatches, ET = _import_openmc_stack()
    except Exception as e:
        return jsonify({'error': f'OpenMC stack not available: {e}'}), 500

    try:
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
        plot_params = {k: v for k, v in plot_params.items() if v not in (None, '')}

        if cross_sections not in CROSS_SECTIONS_LIBRARIES:
            return jsonify({'error': f'Invalid cross-section library: {cross_sections}'}), 400

        selected_cross_sections_path = CROSS_SECTIONS_LIBRARIES[cross_sections]
        if not os.path.exists(selected_cross_sections_path):
            return jsonify({'error': f'Cross-section library not found: {selected_cross_sections_path}'}), 400

        job_dir = os.path.abspath(f'/workspace/jobs/{job_id}')
        os.makedirs(job_dir, exist_ok=True)

        required_files = ['geometry', 'materials', 'settings']
        optional_files = ['tallies', 'depletion']
        file_paths = {}

        for file_type in required_files:
            if file_type not in request.files:
                return jsonify({'error': f'Missing {file_type}.xml file'}), 400
            file = request.files[file_type]
            if file.filename == '':
                return jsonify({'error': f'No {file_type}.xml file selected'}), 400
            fname = f'{file_type}.xml'
            saved = save_uploaded_file(file, job_dir, fname)
            if not saved:
                return jsonify({'error': f'Failed to save {file_type}.xml file'}), 500
            file_paths[file_type] = saved

        for file_type in optional_files:
            if file_type in request.files:
                file = request.files[file_type]
                if file.filename:
                    fname = f'{file_type}.xml' if file_type == 'tallies' else f'{file_type}.ipynb'
                    saved = save_uploaded_file(file, job_dir, fname)
                    if saved:
                        file_paths[file_type] = saved

        # Save to DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        payload = {'file_paths': file_paths, 'plot_params': plot_params}
        c.execute('''INSERT INTO jobs (job_id, email, status, file_paths, job_dir, cross_sections, mode)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (job_id, email, 'queued', json.dumps(payload), job_dir, cross_sections, 'xml'))
        conn.commit()
        conn.close()

        run_simulation.delay(job_id)
        return jsonify({'job_id': job_id}), 202
    except Exception as e:
        logger.exception('Submit job failed')
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------------------
# Code editor job: run user Python to generate XML, then run OpenMC
# ----------------------------------------------------------------------------
@app.route('/api/run-code', methods=['POST'])
def run_code():
    try:
        # Validate input
        data = request.get_json(force=True)
        code_text = data.get('code', '')
        email = data.get('email', 'test@example.com')
        cross_sections = data.get('cross_sections', 'jeff-3.3')
        if not code_text.strip():
            return jsonify({'error': 'Empty code'}), 400

        if cross_sections not in CROSS_SECTIONS_LIBRARIES:
            return jsonify({'error': f'Invalid cross-section library: {cross_sections}'}), 400

        job_id = str(uuid.uuid4())
        job_dir = os.path.abspath(f'/workspace/jobs/{job_id}')
        os.makedirs(job_dir, exist_ok=True)

        # Persist initial DB row with status 'generating'
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO jobs (job_id, email, status, file_paths, job_dir, cross_sections, mode)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (job_id, email, 'generating', json.dumps({}), job_dir, cross_sections, 'code'))
        conn.commit()
        conn.close()

        # Save code and run it in subprocess in job dir
        code_path = os.path.join(job_dir, 'user_code.py')
        log_path = os.path.join(job_dir, 'generation.log')
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(code_text)

        env = os.environ.copy()
        env['OPENMC_CROSS_SECTIONS'] = CROSS_SECTIONS_LIBRARIES[cross_sections]

        # Run code with python in isolated working dir, capture output
        with open(log_path, 'w', encoding='utf-8') as logf:
            proc = subprocess.Popen(
                ['python', 'user_code.py'],
                cwd=job_dir,
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            )
            ret = proc.wait(timeout=int(os.getenv('CODE_TIMEOUT_SEC', '240')))

        # Verify XMLs
        geometry_xml = os.path.join(job_dir, 'geometry.xml')
        materials_xml = os.path.join(job_dir, 'materials.xml')
        settings_xml = os.path.join(job_dir, 'settings.xml')
        tallies_xml = os.path.join(job_dir, 'tallies.xml')

        if not (os.path.exists(geometry_xml) and os.path.exists(materials_xml) and os.path.exists(settings_xml)):
            # Return generation log for debugging
            try:
                with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                    gen_log = f.read()[-20000:]
            except Exception:
                gen_log = ''
            return jsonify({
                'error': 'Code did not generate required XML files (geometry.xml, materials.xml, settings.xml).',
                'job_id': job_id,
                'log_tail': gen_log
            }), 400

        # Save file paths and queue simulation
        file_paths = {
            'geometry': geometry_xml,
            'materials': materials_xml,
            'settings': settings_xml,
        }
        if os.path.exists(tallies_xml):
            file_paths['tallies'] = tallies_xml

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        payload = {'file_paths': file_paths, 'plot_params': {}}
        c.execute('UPDATE jobs SET status=?, file_paths=? WHERE job_id=?', ('queued', json.dumps(payload), job_id))
        conn.commit()
        conn.close()

        # Enqueue OpenMC simulation
        run_simulation.delay(job_id)
        return jsonify({'job_id': job_id, 'status': 'queued'}), 202
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'Code execution timed out'}), 408
    except Exception as e:
        logger.exception('run-code failed')
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/<job_id>')
def fetch_logs(job_id):
    log_path = os.path.abspath(f'/workspace/jobs/{job_id}/generation.log')
    if not os.path.exists(log_path):
        return jsonify({'log': ''})
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()[-50000:]
        return jsonify({'log': content})
    except Exception:
        return jsonify({'log': ''})


# ----------------------------------------------------------------------------
# AI code suggestions (optional free fallback)
# ----------------------------------------------------------------------------
@app.route('/api/suggest', methods=['POST'])
def suggest():
    try:
        data = request.get_json(force=True)
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'suggestion': ''})

        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            import requests
            model_id = os.getenv('HF_MODEL', 'mistralai/Mistral-7B-Instruct-v0.3')
            headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
            payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256, "temperature": 0.2}}
            r = requests.post(f'https://api-inference.huggingface.co/models/{model_id}',
                              headers=headers, json=payload, timeout=30)
            if r.ok:
                try:
                    j = r.json()
                    # HFI API returns a list of generated_text sometimes
                    if isinstance(j, list) and j and 'generated_text' in j[0]:
                        return jsonify({'suggestion': j[0]['generated_text']})
                    # Fallback for text-generation stream style
                    return jsonify({'suggestion': json.dumps(j)[:1000]})
                except Exception:
                    return jsonify({'suggestion': r.text[:1000]})
            else:
                logger.warning('HF API request failed; falling back')

        # Free fallback: simple heuristics and starter snippet
        fallback = (
            "# Starter OpenMC script\n"
            "import openmc\n\n"
            "# Materials\n"
            "water = openmc.Material(name='Water')\n"
            "water.set_density('g/cm3', 1.0)\n"
            "water.add_element('H', 2)\n"
            "water.add_element('O', 1)\n"
            "mats = openmc.Materials([water])\n\n"
            "# Geometry\n"
            "sphere = openmc.Sphere(r=50.0, boundary_type='vacuum')\n"
            "cell = openmc.Cell(region=-sphere, fill=water)\n"
            "geom = openmc.Geometry([cell])\n\n"
            "# Settings\n"
            "settings = openmc.Settings()\n"
            "settings.batches = 50\n"
            "settings.inactive = 10\n"
            "settings.particles = 2000\n\n"
            "# Export XML\n"
            "mats.export_to_xml()\n"
            "geom.export_to_xml()\n"
            "settings.export_to_xml()\n"
        )
        return jsonify({'suggestion': fallback})
    except Exception:
        return jsonify({'suggestion': ''})


# ----------------------------------------------------------------------------
# Simulation task and helpers (trimmed, adapted from user message)
# ----------------------------------------------------------------------------
def analyze_geometry_bounds(geometry_path):
    try:
        openmc, h5py, matplotlib, plt, mpatches, ET = _import_openmc_stack()
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
                except ValueError:
                    pass
        for cylinder in root.findall('.//cylinder'):
            r = cylinder.get('r')
            if r:
                try:
                    rv = float(r)
                    analysis['cylinders'].append(rv)
                    analysis['max_radius'] = max(analysis['max_radius'], rv)
                    if rv > 0:
                        analysis['geometry_type'] = 'cylindrical'
                except ValueError:
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
                    except ValueError:
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
                    except ValueError:
                        pass
        analysis['planes'] = coords
        if analysis['max_radius'] > 0:
            dim = analysis['max_radius'] * 2.4
            width = height = dim
        elif analysis['max_coordinate'] > 0:
            dim = analysis['max_coordinate'] * 2.4
            width = height = dim
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
        res_mul = 0.8
    elif size_factor > 10:
        res_mul = 1.0
    elif size_factor > 1:
        res_mul = 1.5
    else:
        res_mul = 2.0
    if geometry_type in ['cylindrical', 'spherical']:
        res_mul *= 1.2
    if num_materials > 5:
        res_mul *= 1.1
    pixels = int(base_pixels * res_mul)
    return max(400, min(pixels, 2400))


def create_adaptive_geometry_plots(materials, geometry_path, job_dir, plot_params=None):
    try:
        openmc, h5py, matplotlib, plt, mpatches, ET = _import_openmc_stack()
        # Parse inputs
        user_width = user_height = user_resolution = None
        user_basis = 'auto'
        user_origin = (0.0, 0.0, 0.0)
        if plot_params:
            if plot_params.get('width'):
                try:
                    user_width = float(plot_params['width'])
                except ValueError:
                    pass
            if plot_params.get('height'):
                try:
                    user_height = float(plot_params['height'])
                except ValueError:
                    pass
            if plot_params.get('resolution') and plot_params['resolution'] != 'auto':
                try:
                    user_resolution = int(plot_params['resolution'])
                except ValueError:
                    pass
            user_basis = plot_params.get('basis', 'auto')
            try:
                user_origin = (
                    float(plot_params.get('origin_x', 0.0)),
                    float(plot_params.get('origin_y', 0.0)),
                    float(plot_params.get('origin_z', 0.0)),
                )
            except ValueError:
                user_origin = (0.0, 0.0, 0.0)

        if user_width is None or user_height is None:
            auto_w, auto_h, auto_origin, analysis = analyze_geometry_bounds(geometry_path)
            width = user_width if user_width is not None else auto_w
            height = user_height if user_height is not None else auto_h
            origin = user_origin if plot_params else auto_origin
        else:
            width, height, origin = user_width, user_height, user_origin
            analysis = {'geometry_type': 'user_defined'}

        if user_resolution is not None:
            main_res = user_resolution
        else:
            num_materials = len(materials)
            geometry_type = analysis.get('geometry_type', 'unknown')
            main_res = calculate_optimal_resolution(width, height, geometry_type, num_materials)
            side_res = int(main_res * 0.75)

        color_palette = [
            'red', 'blue', 'green', 'orange', 'purple', 'cyan',
            'magenta', 'yellow', 'brown', 'pink', 'gray', 'olive',
            'navy', 'lime', 'maroon', 'teal', 'silver', 'gold'
        ]

        plots_to_create = []
        if user_basis == 'auto':
            plot_xy = openmc.Plot()
            plot_xy.filename = 'geometry_plot'
            plot_xy.width = (width, height)
            plot_xy.pixels = (main_res, main_res)
            plot_xy.color_by = 'material'
            plot_xy.basis = 'xy'
            plot_xy.origin = origin
            plot_xy.colors = {}
            for i, material in enumerate(materials):
                mid = int(material.id)
                plot_xy.colors[mid] = color_palette[i % len(color_palette)]
            plots_to_create.append(plot_xy)

            if (analysis.get('geometry_type') in ['cylindrical', 'box'] or
                analysis.get('max_coordinate', 0) > 0 or len(materials) > 2):
                plot_xz = openmc.Plot()
                plot_xz.filename = 'geometry_plot_xz'
                plot_xz.width = (width, width)
                plot_xz.pixels = (side_res, side_res)
                plot_xz.color_by = 'material'
                plot_xz.basis = 'xz'
                plot_xz.origin = origin
                plot_xz.colors = plot_xy.colors.copy()
                plots_to_create.append(plot_xz)

                if len(materials) > 3 or analysis.get('geometry_type') == 'box':
                    plot_yz = openmc.Plot()
                    plot_yz.filename = 'geometry_plot_yz'
                    plot_yz.width = (height, width)
                    plot_yz.pixels = (side_res, side_res)
                    plot_yz.color_by = 'material'
                    plot_yz.basis = 'yz'
                    plot_yz.origin = origin
                    plot_yz.colors = plot_xy.colors.copy()
                    plots_to_create.append(plot_yz)
        else:
            plot = openmc.Plot()
            plot.filename = f'geometry_plot_{user_basis}'
            plot.width = (width, height)
            plot.pixels = (main_res, main_res)
            plot.color_by = 'material'
            plot.basis = user_basis
            plot.origin = origin
            plot.colors = {}
            for i, material in enumerate(materials):
                mid = int(material.id)
                plot.colors[mid] = color_palette[i % len(color_palette)]
            plots_to_create.append(plot)

        plots = openmc.Plots(plots_to_create)
        plots.export_to_xml()
        try:
            openmc.plot_geometry()
        except Exception as e:
            logger.warning(f"plot_geometry failed: {e}")
            return False, []

        created = []
        for name in ['geometry_plot.png', 'geometry_plot_xz.png', 'geometry_plot_yz.png']:
            if os.path.exists(os.path.join(job_dir, name)):
                created.append(os.path.join(job_dir, name))
        return True, created
    except Exception as e:
        logger.exception('create_adaptive_geometry_plots failed')
        return False, []


@celery.task
def run_simulation(job_id):
    conn = None
    original_dir = os.getcwd()
    try:
        import traceback
        openmc, h5py, matplotlib, plt, mpatches, ET = _import_openmc_stack()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT file_paths, job_dir, cross_sections FROM jobs WHERE job_id = ?", (job_id,))
        row = c.fetchone()
        if not row:
            raise RuntimeError(f'Job {job_id} not found')
        file_paths_json, job_dir, cross_sections = row
        try:
            job_data = json.loads(file_paths_json) if file_paths_json else {}
            if isinstance(job_data, dict) and 'file_paths' in job_data:
                file_paths = job_data['file_paths']
                plot_params = job_data.get('plot_params', {})
            else:
                file_paths = job_data
                plot_params = {}
        except Exception as e:
            raise RuntimeError(f'Invalid file_paths data: {e}')

        if not os.path.exists(job_dir):
            raise RuntimeError(f'Job directory missing: {job_dir}')

        for req in ['geometry', 'materials', 'settings']:
            if req not in file_paths or not os.path.exists(file_paths[req]):
                raise RuntimeError(f'Missing required XML: {req}')

        os.chdir(job_dir)
        os.environ['OPENMC_CROSS_SECTIONS'] = CROSS_SECTIONS_LIBRARIES[cross_sections]
        c.execute('UPDATE jobs SET status=? WHERE job_id=?', ('running', job_id))
        conn.commit()

        materials = openmc.Materials.from_xml(file_paths['materials'])
        geometry = openmc.Geometry.from_xml(file_paths['geometry'])
        settings = openmc.Settings.from_xml(file_paths['settings'])
        model = openmc.Model(geometry=geometry, materials=materials, settings=settings)
        if 'tallies' in file_paths and os.path.exists(file_paths['tallies']):
            tallies = openmc.Tallies.from_xml(file_paths['tallies'])
            model.tallies = tallies

        model.run()

        # Try to create geometry plots
        try:
            create_adaptive_geometry_plots(materials, file_paths['geometry'], job_dir, plot_params)
        except Exception:
            logger.warning('Plotting failed (ignored)')

        c.execute('UPDATE jobs SET status=? WHERE job_id=?', ('completed', job_id))
        conn.commit()
    except Exception as e:
        if conn:
            c = conn.cursor()
            c.execute('UPDATE jobs SET status=? WHERE job_id=?', (f'failed: {e}', job_id))
            conn.commit()
        raise
    finally:
        if conn:
            conn.close()
        os.chdir(original_dir)


@app.route('/status/<job_id>')
def get_status(job_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT status FROM jobs WHERE job_id=?', (job_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return jsonify({'status': row[0]})
    return jsonify({'error': 'Job not found'}), 404


# Results page (trimmed variant using statepoint summarization best-effort)
@app.route('/results/<job_id>')
def show_results(job_id):
    try:
        openmc, h5py, matplotlib, plt, mpatches, ET = _import_openmc_stack()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT status, job_dir, cross_sections, file_paths FROM jobs WHERE job_id=?', (job_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return render_template('results.html', job_id=job_id, status='Job not found', results=None, plot_data=None, geometry_views={}, cross_sections_used=None)

        status, job_dir, cross_sections_used, file_paths_json = row
        try:
            job_data = json.loads(file_paths_json) if file_paths_json else {}
            file_paths = job_data['file_paths'] if isinstance(job_data, dict) and 'file_paths' in job_data else job_data
        except Exception:
            file_paths = {}

        results = []
        plot_data = None
        geometry_views = {}

        if status == 'completed' and os.path.exists(job_dir):
            statepoints = [f for f in os.listdir(job_dir) if f.startswith('statepoint') and f.endswith('.h5')]
            if statepoints:
                sp_path = os.path.join(job_dir, sorted(statepoints)[-1])
                try:
                    sp = openmc.StatePoint(sp_path)
                    if hasattr(sp, 'keff'):
                        results.append({'metric': 'k-effective', 'value': f"{sp.keff.nominal_value:.6f}", 'uncertainty': f"± {sp.keff.std_dev:.6f}"})
                except Exception as _:
                    pass

            # Attach any generated geometry plots
            for nm in ['geometry_plot.png', 'geometry_plot_xz.png', 'geometry_plot_yz.png']:
                p = os.path.join(job_dir, nm)
                if os.path.exists(p):
                    with open(p, 'rb') as f:
                        geometry_views[nm.replace('.png', '')] = base64.b64encode(f.read()).decode('utf-8')

        return render_template('results.html', job_id=job_id, status=status, results=results, plot_data=plot_data, geometry_views=geometry_views, cross_sections_used=cross_sections_used, file_paths=file_paths, job_dir=job_dir)
    except Exception as e:
        logger.exception('show_results failed')
        return render_template('results.html', job_id=job_id, status=f'Error: {e}', results=[{'metric': 'System Error', 'value': f'{e}', 'uncertainty': 'Contact support'}], plot_data=None, geometry_views={}, cross_sections_used=None, file_paths={}, job_dir=None)


@app.route('/cross-sections-info')
def cross_sections_info():
    info = {}
    for name, path in CROSS_SECTIONS_LIBRARIES.items():
        info[name] = {'path': path, 'exists': os.path.exists(path), 'size': os.path.getsize(path) if os.path.exists(path) else 0}
    return jsonify(info)


# ----------------------------------------------------------------------------
# Entry
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    os.makedirs('/workspace/jobs', exist_ok=True)
    init_db()
    missing = [f"{k}: {v}" for k, v in CROSS_SECTIONS_LIBRARIES.items() if not os.path.exists(v)]
    if missing:
        print('WARNING: Missing cross-sections:')
        for m in missing:
            print('  -', m)
        print('Simulations will fail if an unavailable library is chosen.')
    app.run(debug=True, host='0.0.0.0', port=int(os.getenv('PORT', '5000')))

