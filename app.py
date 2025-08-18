from flask import Flask, request, jsonify, render_template
from celery import Celery
import sqlite3
import os
import uuid
import openmc
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import base64
from io import BytesIO
import json
import logging
import xml.etree.ElementTree as ET
from werkzeug.utils import secure_filename
import subprocess
import shlex
import time
import threading
import requests

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.before_request
def before_request():
    pass


@app.after_request
def after_request(response):
    response.headers['ngrok-skip-browser-warning'] = 'any'
    return response


# Celery configuration
app.config['CELERY_BROKER_URL'] = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
app.config['CELERY_RESULT_BACKEND'] = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Initialize Celery
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


# Cross-sections paths - UPDATE THESE PATHS TO MATCH YOUR SYSTEM
CROSS_SECTIONS_LIBRARIES = {
    'jeff-3.3': os.environ.get('OPENMC_XS_JEFF33', '/home/omio/Code/OpenMc/jeff-3.3-hdf5/cross_sections.xml'),
    'endfb-viii.0': os.environ.get('OPENMC_XS_ENDFBVIII', '/home/omio/Code/OpenMc/endfb-viii.0-hdf5/cross_sections.xml'),
    'endfb-vii.1': os.environ.get('OPENMC_XS_ENDFBVII1', '/home/omio/Code/OpenMc/endfb-vii.1-hdf5/cross_sections.xml')
}


def analyze_geometry_bounds(geometry_path):
    try:
        tree = ET.parse(geometry_path)
        root = tree.getroot()

        analysis = {
            'spheres': [],
            'cylinders': [],
            'planes': [],
            'boxes': [],
            'max_radius': 0.0,
            'max_coordinate': 0.0,
            'geometry_type': 'unknown'
        }

        for sphere in root.findall('.//sphere'):
            radius_attr = sphere.get('r')
            if radius_attr:
                try:
                    radius = float(radius_attr)
                    analysis['spheres'].append(radius)
                    analysis['max_radius'] = max(analysis['max_radius'], radius)
                    if radius > 0:
                        analysis['geometry_type'] = 'spherical'
                except ValueError:
                    pass

        for cylinder in root.findall('.//cylinder'):
            radius_attr = cylinder.get('r')
            if radius_attr:
                try:
                    radius = float(radius_attr)
                    analysis['cylinders'].append(radius)
                    analysis['max_radius'] = max(analysis['max_radius'], radius)
                    if radius > 0:
                        analysis['geometry_type'] = 'cylindrical'
                except ValueError:
                    pass

        coords = []
        for plane in root.findall('.//plane'):
            for attr in ['x0', 'y0', 'z0']:
                coord_attr = plane.get(attr)
                if coord_attr:
                    try:
                        coord = abs(float(coord_attr))
                        coords.append(coord)
                        analysis['max_coordinate'] = max(analysis['max_coordinate'], coord)
                    except ValueError:
                        pass

        for box in root.findall('.//box'):
            for attr in ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']:
                coord_attr = box.get(attr)
                if coord_attr:
                    try:
                        coord = abs(float(coord_attr))
                        coords.append(coord)
                        analysis['max_coordinate'] = max(analysis['max_coordinate'], coord)
                        analysis['geometry_type'] = 'box'
                    except ValueError:
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

        width = max(width, 1.0)
        height = max(height, 1.0)
        width = min(width, 1000.0)
        height = min(height, 1000.0)

        origin = (0.0, 0.0, 0.0)
        return width, height, origin, analysis

    except Exception as e:
        logger.warning(f"Error analyzing geometry bounds: {e}")
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
    try:
        logger.info("Generating adaptive geometry plots...")
        user_width = None
        user_height = None
        user_resolution = None
        user_basis = 'auto'
        user_origin = (0.0, 0.0, 0.0)

        if plot_params:
            if plot_params.get('width'):
                try:
                    user_width = float(plot_params['width'])
                except ValueError:
                    logger.warning(f"Invalid width: {plot_params['width']}")

            if plot_params.get('height'):
                try:
                    user_height = float(plot_params['height'])
                except ValueError:
                    logger.warning(f"Invalid height: {plot_params['height']}")

            if plot_params.get('resolution') and plot_params['resolution'] != 'auto':
                try:
                    user_resolution = int(plot_params['resolution'])
                except ValueError:
                    logger.warning(f"Invalid resolution: {plot_params['resolution']}")

            user_basis = plot_params.get('basis', 'auto')

            try:
                origin_x = float(plot_params.get('origin_x', 0.0))
                origin_y = float(plot_params.get('origin_y', 0.0))
                origin_z = float(plot_params.get('origin_z', 0.0))
                user_origin = (origin_x, origin_y, origin_z)
            except ValueError:
                logger.warning("Invalid origin values, using default (0,0,0)")
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
            side_resolution = int(main_resolution * 0.75)
        else:
            num_materials = len(materials)
            geometry_type = analysis.get('geometry_type', 'unknown')
            main_resolution = calculate_optimal_resolution(width, height, geometry_type, num_materials)
            side_resolution = int(main_resolution * 0.75)

        logger.info(f"Using dimensions: {width:.2f}x{height:.2f}, resolution: {main_resolution}")
        logger.info(f"Origin: {origin}, basis preference: {user_basis}")

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
            plot_xy.pixels = (main_resolution, main_resolution)
            plot_xy.color_by = 'material'
            plot_xy.basis = 'xy'
            plot_xy.origin = origin

            plot_xy.colors = {}
            for i, material in enumerate(materials):
                material_id = int(material.id)
                plot_xy.colors[material_id] = color_palette[i % len(color_palette)]

            plots_to_create.append(plot_xy)

            if len(materials) > 2:
                plot_xz = openmc.Plot()
                plot_xz.filename = 'geometry_plot_xz'
                plot_xz.width = (width, width)
                plot_xz.pixels = (side_resolution, side_resolution)
                plot_xz.color_by = 'material'
                plot_xz.basis = 'xz'
                plot_xz.origin = origin
                plot_xz.colors = plot_xy.colors.copy()
                plots_to_create.append(plot_xz)

                plot_yz = openmc.Plot()
                plot_yz.filename = 'geometry_plot_yz'
                plot_yz.width = (height, width)
                plot_yz.pixels = (side_resolution, side_resolution)
                plot_yz.color_by = 'material'
                plot_yz.basis = 'yz'
                plot_yz.origin = origin
                plot_yz.colors = plot_xy.colors.copy()
                plots_to_create.append(plot_yz)
        else:
            plot = openmc.Plot()
            plot.filename = f'geometry_plot_{user_basis}'
            plot.width = (width, height)
            plot.pixels = (main_resolution, main_resolution)
            plot.color_by = 'material'
            plot.basis = user_basis
            plot.origin = origin

            plot.colors = {}
            for i, material in enumerate(materials):
                material_id = int(material.id)
                plot.colors[material_id] = color_palette[i % len(color_palette)]

            plots_to_create.append(plot)

        plots = openmc.Plots(plots_to_create)
        plots.export_to_xml()

        try:
            openmc.plot_geometry()
        except Exception as plot_error:
            logger.error(f"Plotting failed: {plot_error}")
            return False, []

        created_plots = []
        plot_files_to_check = [
            'geometry_plot.png', 'geometry_plot_xz.png',
            'geometry_plot_yz.png'
        ]

        for plot_file in plot_files_to_check:
            if os.path.exists(plot_file):
                created_plots.append(plot_file)

        enhanced_plots = []
        for plot_file in created_plots:
            try:
                image = plt.imread(plot_file)
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.imshow(image)
                if '_xz' in plot_file:
                    xlabel = 'X (cm)'; ylabel = 'Z (cm)'
                elif '_yz' in plot_file:
                    xlabel = 'Y (cm)'; ylabel = 'Z (cm)'
                else:
                    xlabel = 'X (cm)'; ylabel = 'Y (cm)'
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                handles = []
                for i, material in enumerate(materials):
                    material_id = int(material.id)
                    color = 'black'
                    if 'plot_xy' in locals():
                        color = plot_xy.colors.get(material_id, 'black')
                    label = material.name if material.name else f'Material {material_id}'
                    handles.append(mpatches.Patch(color=color, label=label))
                if handles:
                    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
                enhanced_file = plot_file.replace('.png', '_enhanced.png')
                plt.savefig(enhanced_file, bbox_inches='tight')
                plt.close()
                enhanced_plots.append(enhanced_file)
            except Exception as enhance_error:
                logger.warning(f"Failed to enhance {plot_file}: {enhance_error}")
                enhanced_plots.append(plot_file)

        if enhanced_plots:
            return True, enhanced_plots
        return False, []
    except Exception as plot_error:
        logger.error(f"Could not generate adaptive geometry plots: {plot_error}")
        return False, []


def init_db():
    conn = sqlite3.connect('jobs.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS jobs
                 (job_id TEXT PRIMARY KEY,
                  email TEXT,
                  status TEXT,
                  file_paths TEXT,
                  job_dir TEXT,
                  cross_sections TEXT DEFAULT 'jeff-3.3')''')
    c.execute("PRAGMA table_info(jobs)")
    columns = [column[1] for column in c.fetchall()]

    if 'cross_sections' not in columns:
        try:
            logger.info("Adding cross_sections column to existing jobs table...")
            c.execute("ALTER TABLE jobs ADD COLUMN cross_sections TEXT DEFAULT 'jeff-3.3'")
        except sqlite3.Error as e:
            logger.error(f"Failed to add cross_sections column: {e}")

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def save_uploaded_file(file, job_dir, filename):
    try:
        secure_name = secure_filename(filename)
        file_path = os.path.join(job_dir, secure_name)
        os.makedirs(job_dir, exist_ok=True)
        file.save(file_path)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logger.info(f"Successfully saved {secure_name} to {file_path} (size: {os.path.getsize(file_path)} bytes)")
            return file_path
        else:
            logger.error(f"File {secure_name} was not saved properly or is empty")
            return None
    except Exception as e:
        logger.error(f"Error saving file {filename}: {str(e)}")
        return None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/editor')
def editor():
    session_id = str(uuid.uuid4())
    job_dir = os.path.abspath(os.path.join('jobs', session_id))
    os.makedirs(job_dir, exist_ok=True)
    return render_template('editor.html', session_id=session_id)


@app.route('/submit', methods=['POST'])
def submit_job():
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
        plot_params = {k: v for k, v in plot_params.items() if v and v != ''}

        if cross_sections not in CROSS_SECTIONS_LIBRARIES:
            return jsonify({'error': f'Invalid cross-section library: {cross_sections}'}), 400

        selected_cross_sections_path = CROSS_SECTIONS_LIBRARIES[cross_sections]
        if not os.path.exists(selected_cross_sections_path):
            return jsonify({'error': f'Cross-section library not found: {selected_cross_sections_path}'}), 400

        job_dir = os.path.abspath(f'jobs/{job_id}')
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
            filename = f'{file_type}.xml'
            saved_path = save_uploaded_file(file, job_dir, filename)
            if saved_path is None:
                return jsonify({'error': f'Failed to save {file_type}.xml file'}), 500
            file_paths[file_type] = saved_path

        for file_type in optional_files:
            if file_type in request.files:
                file = request.files[file_type]
                if file.filename != '':
                    filename = f'{file_type}.xml' if file_type == 'tallies' else f'{file_type}.ipynb'
                    saved_path = save_uploaded_file(file, job_dir, filename)
                    if saved_path is not None:
                        file_paths[file_type] = saved_path

        for file_type in required_files:
            file_path = file_paths[file_type]
            if not os.path.exists(file_path):
                logger.error(f"Required file {file_type}.xml does not exist at {file_path}")
                return jsonify({'error': f'Failed to save {file_type}.xml file properly'}), 500

        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        job_data = {
            'file_paths': file_paths,
            'plot_params': plot_params
        }
        c.execute('''INSERT INTO jobs (job_id, email, status, file_paths, job_dir, cross_sections)
                     VALUES (?, ?, ?, ?, ?, ?)''',
                  (job_id, email, 'queued', json.dumps(job_data), job_dir, cross_sections))
        conn.commit()
        conn.close()

        run_simulation.delay(job_id)
        return jsonify({'job_id': job_id}), 202

    except Exception as e:
        logger.error(f"Error submitting job: {str(e)}")
        return jsonify({'error': str(e)}), 500


def _write_code_to_file(code_text, job_dir):
    main_path = os.path.join(job_dir, 'main.py')
    with open(main_path, 'w') as f:
        f.write(code_text)
    return main_path


def _call_ollama(prompt, model=None, temperature=0.2, max_tokens=512):
    try:
        ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        if not model:
            model = os.environ.get('OLLAMA_MODEL', 'qwen2.5-coder:7b')
        url = f"{ollama_host.rstrip('/')}/api/generate"
        resp = requests.post(url, json={
            'model': model,
            'prompt': prompt,
            'temperature': temperature,
            'stream': False
        }, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            return True, data.get('response', '')
        return False, f"Ollama error {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)


@app.route('/api/editor/suggest', methods=['POST'])
def ai_suggest():
    payload = request.get_json(silent=True) or {}
    context = payload.get('context', '')
    question = payload.get('question', '')
    system_hint = (
        "You are an assistant helping to write Python code that builds OpenMC simulations. "
        "Prioritize producing valid Python using openmc.Materials, openmc.Geometry, openmc.Settings. "
        "Output only code blocks without explanations."
    )
    prompt = f"{system_hint}\n\nUser question:\n{question}\n\nProject context:\n{context}\n\nProvide the best possible code answer."

    ok, resp = _call_ollama(prompt)
    if ok:
        return jsonify({'success': True, 'suggestion': resp})
    return jsonify({'success': False, 'error': resp}), 200


@app.route('/api/editor/start', methods=['POST'])
def start_editor_session():
    session_id = str(uuid.uuid4())
    job_dir = os.path.abspath(os.path.join('jobs', session_id))
    os.makedirs(job_dir, exist_ok=True)
    return jsonify({'session_id': session_id, 'job_dir': job_dir})


@app.route('/api/editor/execute', methods=['POST'])
def execute_code():
    try:
        data = request.get_json(silent=True) or {}
        session_id = data.get('session_id')
        code_text = data.get('code', '')
        auto_run_openmc = bool(data.get('auto_run_openmc', False))
        cross_sections = data.get('cross_sections', 'jeff-3.3')

        if not session_id or not code_text:
            return jsonify({'error': 'session_id and code are required'}), 400

        job_dir = os.path.abspath(os.path.join('jobs', session_id))
        os.makedirs(job_dir, exist_ok=True)

        execution_id = str(uuid.uuid4())
        exec_meta = {
            'session_id': session_id,
            'execution_id': execution_id,
            'status': 'queued'
        }

        execute_user_code.delay(session_id, execution_id, code_text, auto_run_openmc, cross_sections)
        return jsonify(exec_meta), 202
    except Exception as e:
        logger.exception("execute_code failed")
        return jsonify({'error': str(e)}), 500


@app.route('/api/editor/result/<execution_id>')
def get_execution_result(execution_id):
    result_path = os.path.abspath(os.path.join('jobs', f'{execution_id}.json'))
    if not os.path.exists(result_path):
        return jsonify({'status': 'pending'})
    try:
        with open(result_path, 'r') as f:
            payload = json.load(f)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'status': 'failed', 'error': str(e)})


@celery.task
def execute_user_code(session_id, execution_id, code_text, auto_run_openmc, cross_sections):
    result_payload = {
        'execution_id': execution_id,
        'session_id': session_id,
        'status': 'running',
        'stdout': '',
        'stderr': '',
        'generated_files': [],
        'openmc': None
    }
    result_path = os.path.abspath(os.path.join('jobs', f'{execution_id}.json'))

    def persist():
        try:
            with open(result_path, 'w') as f:
                json.dump(result_payload, f)
        except Exception:
            pass

    job_dir = os.path.abspath(os.path.join('jobs', session_id))
    os.makedirs(job_dir, exist_ok=True)
    main_path = _write_code_to_file(code_text, job_dir)

    env = os.environ.copy()
    try:
        xs_path = CROSS_SECTIONS_LIBRARIES.get(cross_sections)
        if xs_path:
            env['OPENMC_CROSS_SECTIONS'] = xs_path
    except Exception:
        pass

    try:
        proc = subprocess.Popen(
            [shlex.which('python') or 'python', '-u', main_path],
            cwd=job_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1
        )
        stdout_lines = []
        stderr_lines = []

        def _read_stream(stream, sink_list, key):
            for line in iter(stream.readline, ''):
                sink_list.append(line)
                result_payload[key] = ''.join(sink_list)[-200000:]
                persist()
            stream.close()

        t_out = threading.Thread(target=_read_stream, args=(proc.stdout, stdout_lines, 'stdout'))
        t_err = threading.Thread(target=_read_stream, args=(proc.stderr, stderr_lines, 'stderr'))
        t_out.start(); t_err.start()
        proc.wait()
        t_out.join(); t_err.join()

        result_payload['return_code'] = proc.returncode
        result_payload['status'] = 'completed' if proc.returncode == 0 else 'failed'

        # Detect generated XML files
        for name in ['geometry.xml', 'materials.xml', 'settings.xml', 'tallies.xml']:
            path = os.path.join(job_dir, name)
            if os.path.exists(path):
                result_payload['generated_files'].append(name)

        # Optionally run OpenMC after code
        if auto_run_openmc and all(os.path.exists(os.path.join(job_dir, f'{n}.xml')) for n in ['geometry', 'materials', 'settings']):
            try:
                materials = openmc.Materials.from_xml(os.path.join(job_dir, 'materials.xml'))
                geometry = openmc.Geometry.from_xml(os.path.join(job_dir, 'geometry.xml'))
                settings = openmc.Settings.from_xml(os.path.join(job_dir, 'settings.xml'))
                model = openmc.Model(geometry=geometry, materials=materials, settings=settings)
                model.run(cwd=job_dir)
                success, geometry_plots = create_adaptive_geometry_plots(materials, os.path.join(job_dir, 'geometry.xml'), job_dir, {})
                result_payload['openmc'] = {
                    'success': True,
                    'plots': geometry_plots
                }
            except Exception as e:
                result_payload['openmc'] = {
                    'success': False,
                    'error': str(e)
                }

    except Exception as e:
        result_payload['status'] = 'failed'
        result_payload['stderr'] += f"\nExecutor error: {e}"
    finally:
        persist()


@celery.task
def run_simulation(job_id):
    conn = None
    original_dir = os.getcwd()
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute("SELECT file_paths, job_dir, cross_sections FROM jobs WHERE job_id = ?", (job_id,))
        result = c.fetchone()
        if not result:
            raise ValueError(f"Job {job_id} not found in database")

        file_paths_json, job_dir, cross_sections = result
        try:
            job_data = json.loads(file_paths_json)
            if isinstance(job_data, dict) and 'file_paths' in job_data:
                file_paths = job_data['file_paths']
                plot_params = job_data.get('plot_params', {})
            else:
                file_paths = job_data
                plot_params = {}
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid file paths data in database: {e}")

        if not os.path.exists(job_dir):
            raise ValueError(f"Job directory does not exist: {job_dir}")

        required_files = ['geometry', 'materials', 'settings']
        for file_type in required_files:
            if file_type not in file_paths:
                raise ValueError(f"Missing file path for {file_type}")
            file_path = file_paths[file_type]
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Required file {file_type}.xml not found at {file_path}")
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValueError(f"File {file_type}.xml is empty: {file_path}")

        os.chdir(job_dir)

        cross_sections_path = CROSS_SECTIONS_LIBRARIES[cross_sections]
        os.environ['OPENMC_CROSS_SECTIONS'] = cross_sections_path

        c.execute("UPDATE jobs SET status = ? WHERE job_id = ?", ('running', job_id))
        conn.commit()

        try:
            materials = openmc.Materials.from_xml(file_paths['materials'])
            geometry = openmc.Geometry.from_xml(file_paths['geometry'])
            settings = openmc.Settings.from_xml(file_paths['settings'])
            model = openmc.Model(geometry=geometry, materials=materials, settings=settings)
            if 'tallies' in file_paths and os.path.exists(file_paths['tallies']):
                tallies = openmc.Tallies.from_xml(file_paths['tallies'])
                model.tallies = tallies
        except Exception as model_error:
            raise RuntimeError(f"Failed to load OpenMC models: {str(model_error)}")

        try:
            model.run()
        except Exception as sim_error:
            raise RuntimeError(f"OpenMC simulation failed: {str(sim_error)}")

        try:
            success, geometry_plots = create_adaptive_geometry_plots(
                materials, file_paths['geometry'], job_dir, plot_params
            )
        except Exception:
            pass

        c.execute("UPDATE jobs SET status = ? WHERE job_id = ?", ('completed', job_id))
        conn.commit()
    except Exception as e:
        try:
            if conn:
                c = conn.cursor()
                c.execute("UPDATE jobs SET status = ? WHERE job_id = ?", (f'failed: {str(e)}', job_id))
                conn.commit()
        except Exception:
            pass
        raise
    finally:
        os.chdir(original_dir)
        if conn:
            conn.close()


@app.route('/status/<job_id>')
def get_status(job_id):
    conn = sqlite3.connect('jobs.db')
    c = conn.cursor()
    c.execute("SELECT status FROM jobs WHERE job_id = ?", (job_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return jsonify({'status': result[0]})
    else:
        return jsonify({'error': 'Job not found'}), 404


@app.route('/results/<job_id>')
def show_results(job_id):
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute("SELECT status, job_dir, cross_sections, file_paths FROM jobs WHERE job_id = ?", (job_id,))
        result = c.fetchone()
        conn.close()

        if not result:
            return render_template('results.html',
                                 job_id=job_id,
                                 status='Job not found',
                                 results=None,
                                 plot_data=None,
                                 geometry_views={},
                                 cross_sections_used=None)

        status, job_dir, cross_sections_used, file_paths_json = result

        try:
            file_paths = json.loads(file_paths_json) if file_paths_json else {}
        except json.JSONDecodeError:
            file_paths = {}

        results = []
        plot_data = None
        geometry_views = {}

        if status == 'completed' and os.path.exists(job_dir):
            try:
                statepoint_files = [f for f in os.listdir(job_dir) if f.startswith('statepoint') and f.endswith('.h5')]
                if statepoint_files:
                    statepoint_files.sort()
                    statepoint_file = os.path.join(job_dir, statepoint_files[-1])
                    try:
                        try:
                            sp = openmc.StatePoint(statepoint_file)
                            if hasattr(sp, 'keff'):
                                keff_mean = sp.keff.nominal_value
                                keff_std = sp.keff.std_dev
                                results.append({'metric': 'k-effective', 'value': f"{keff_mean:.6f}", 'uncertainty': f"± {keff_std:.6f}"})
                            elif hasattr(sp, 'k_combined'):
                                keff_data = sp.k_combined
                                results.append({'metric': 'k-effective (combined)', 'value': f"{keff_data.nominal_value:.6f}", 'uncertainty': f"± {keff_data.std_dev:.6f}"})
                            if hasattr(sp, 'n_batches'):
                                results.append({'metric': 'Number of batches', 'value': f"{sp.n_batches}", 'uncertainty': 'N/A'})
                            if hasattr(sp, 'n_inactive'):
                                results.append({'metric': 'Inactive batches', 'value': f"{sp.n_inactive}", 'uncertainty': 'N/A'})
                            if hasattr(sp, 'n_particles'):
                                results.append({'metric': 'Particles per batch', 'value': f"{sp.n_particles}", 'uncertainty': 'N/A'})
                            if hasattr(sp, 'runtime'):
                                runtime_data = sp.runtime
                                if isinstance(runtime_data, (list, tuple)) and len(runtime_data) > 0:
                                    total_runtime = runtime_data[0]
                                    results.append({'metric': 'Total runtime (seconds)', 'value': f"{total_runtime:.3f}", 'uncertainty': 'N/A'})
                                elif isinstance(runtime_data, (int, float)):
                                    results.append({'metric': 'Total runtime (seconds)', 'value': f"{runtime_data:.3f}", 'uncertainty': 'N/A'})
                            if sp.tallies:
                                tally_means = []
                                tally_ids = []
                                for tally_id, tally in sp.tallies.items():
                                    try:
                                        mean_val = tally.mean.flatten()[0] if hasattr(tally, 'mean') else 0.0
                                        std_val = tally.std_dev.flatten()[0] if hasattr(tally, 'std_dev') else 0.0
                                        results.append({'metric': f'Tally {tally_id}', 'value': f"{mean_val:.6e}", 'uncertainty': f"± {std_val:.6e}"})
                                        tally_means.append(mean_val)
                                        tally_ids.append(tally_id)
                                    except Exception as tally_error:
                                        results.append({'metric': f'Tally {tally_id}', 'value': 'Could not extract', 'uncertainty': f'Error: {str(tally_error)}'})
                                if tally_means:
                                    plt.figure(figsize=(12, 8))
                                    bars = plt.bar([f"Tally {tid}" for tid in tally_ids], tally_means,
                                                   color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe'])
                                    plt.xlabel('Tally ID', fontsize=14, fontweight='bold')
                                    plt.ylabel('Mean Value', fontsize=14, fontweight='bold')
                                    plt.title('Tally Results from OpenMC Simulation', fontsize=16, fontweight='bold')
                                    plt.xticks(rotation=45)
                                    plt.grid(True, alpha=0.3)
                                    for bar, value in zip(bars, tally_means):
                                        height = bar.get_height()
                                        plt.text(bar.get_x() + bar.get_width()/2., height, f'{value:.2e}', ha='center', va='bottom')
                                    plt.tight_layout()
                                    img_buffer = BytesIO()
                                    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=150)
                                    img_buffer.seek(0)
                                    plot_data = base64.b64encode(img_buffer.getvalue()).decode()
                                    plt.close()
                        except Exception:
                            with h5py.File(statepoint_file, 'r') as f:
                                if 'keff' in f:
                                    try:
                                        keff_data = f['keff'][()]
                                        if len(keff_data) >= 2:
                                            results.append({'metric': 'k-effective', 'value': f"{keff_data[0]:.6f}", 'uncertainty': f"± {keff_data[1]:.6f}"})
                                    except Exception:
                                        pass
                                for param in ['n_batches', 'n_inactive', 'n_particles']:
                                    if param in f:
                                        try:
                                            value = int(f[param][()])
                                            readable_name = param.replace('_', ' ').title()
                                            results.append({'metric': readable_name, 'value': f"{value}", 'uncertainty': 'N/A'})
                                        except Exception:
                                            pass
                                if 'runtime' in f:
                                    try:
                                        runtime_data = f['runtime']
                                        if hasattr(runtime_data, '__len__') and len(runtime_data) > 0:
                                            total_runtime = runtime_data[0]
                                            results.append({'metric': 'Total runtime (seconds)', 'value': f"{total_runtime:.3f}", 'uncertainty': 'N/A'})
                                    except Exception:
                                        pass
                    except Exception as hdf5_error:
                        results.append({'metric': 'File Read Error', 'value': f'{str(hdf5_error)}', 'uncertainty': 'Check file integrity'})
                else:
                    try:
                        all_files = os.listdir(job_dir)
                        results.append({'metric': 'File Status', 'value': 'No statepoint files found', 'uncertainty': f"Files available: {', '.join(all_files)}"})
                    except Exception as list_error:
                        results.append({'metric': 'Directory Error', 'value': 'Cannot access result directory', 'uncertainty': str(list_error)})
            except Exception as parse_error:
                results.append({'metric': 'Parse Error', 'value': f'{str(parse_error)}', 'uncertainty': 'Contact support'})

        if job_dir and os.path.exists(job_dir):
            try:
                original_cwd = os.getcwd()
                os.chdir(job_dir)
                plot_patterns = [
                    ('xy', ['geometry_plot_enhanced.png', 'geometry_plot.png']),
                    ('xz', ['geometry_plot_xz_enhanced.png', 'geometry_plot_xz.png']),
                    ('yz', ['geometry_plot_yz_enhanced.png', 'geometry_plot_yz.png']),
                    ('xy_zoom', ['geometry_plot_zoom_enhanced.png', 'geometry_plot_zoom.png']),
                    ('simple', ['geometry_plot_simple.png', 'geometry_plot_minimal.png'])
                ]
                for view_type, file_patterns in plot_patterns:
                    if view_type in geometry_views:
                        continue
                    for pattern in file_patterns:
                        if os.path.exists(pattern):
                            try:
                                with open(pattern, 'rb') as f:
                                    image_data = f.read()
                                    geometry_views[view_type] = base64.b64encode(image_data).decode('utf-8')
                                    break
                            except Exception:
                                continue
            finally:
                os.chdir(original_cwd)

        return render_template('results.html',
                             job_id=job_id,
                             status=status,
                             results=results,
                             plot_data=plot_data,
                             geometry_views=geometry_views,
                             cross_sections_used=cross_sections_used,
                             file_paths=file_paths,
                             job_dir=job_dir)

    except Exception as e:
        return render_template('results.html',
                             job_id=job_id,
                             status=f'Error: {str(e)}',
                             results=[{'metric': 'System Error', 'value': f'{str(e)}', 'uncertainty': 'Contact support'}],
                             plot_data=None,
                             geometry_views={},
                             cross_sections_used=None,
                             file_paths={},
                             job_dir=None)


@app.route('/generate-manual-plot/<job_id>')
def generate_manual_plot(job_id):
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
        c.execute("SELECT file_paths, job_dir FROM jobs WHERE job_id = ?", (job_id,))
        result = c.fetchone()
        conn.close()

        if not result:
            return jsonify({'error': 'Job not found'}), 404

        file_paths_json, job_dir = result
        try:
            file_paths = json.loads(file_paths_json) if file_paths_json else {}
        except json.JSONDecodeError:
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

            color_palette = [
                'red', 'blue', 'green', 'orange', 'purple', 'cyan',
                'magenta', 'yellow', 'brown', 'pink', 'gray', 'olive',
                'navy', 'lime', 'maroon', 'teal', 'silver', 'gold'
            ]
            plot.colors = {}
            try:
                for i, material in enumerate(materials):
                    material_id = int(material.id)
                    if material_id > 0:
                        plot.colors[material_id] = color_palette[i % len(color_palette)]
            except Exception as color_error:
                logger.warning(f"Error setting manual plot colors: {color_error}")

            plots = openmc.Plots([plot])
            plots.export_to_xml()
            openmc.plot_geometry()

            plot_file = f'{plot_filename}.png'
            if os.path.exists(plot_file):
                with open(plot_file, 'rb') as f:
                    plot_data = base64.b64encode(f.read()).decode()
                return jsonify({'success': True, 'plot_data': plot_data, 'parameters': {'width': width, 'height': height, 'basis': basis, 'resolution': resolution, 'origin': [origin_x, origin_y, origin_z]}})
            else:
                return jsonify({'error': 'Failed to generate manual plot'}), 500
        finally:
            os.chdir(original_dir)
    except Exception as e:
        logger.error(f"Error generating manual plot: {str(e)}")
        return jsonify({'error': f'Failed to generate manual plot: {str(e)}'}), 500


@app.route('/cross-sections-info')
def cross_sections_info():
    library_info = {}
    for lib_name, lib_path in CROSS_SECTIONS_LIBRARIES.items():
        library_info[lib_name] = {
            'path': lib_path,
            'exists': os.path.exists(lib_path),
            'size': os.path.getsize(lib_path) if os.path.exists(lib_path) else 0
        }
    return jsonify(library_info)


@app.route('/geometry-preview/<job_id>')
def geometry_preview(job_id):
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute("SELECT file_paths, job_dir FROM jobs WHERE job_id = ?", (job_id,))
        result = c.fetchone()
        conn.close()
        if not result:
            return jsonify({'error': 'Job not found'}), 404
        file_paths_json, job_dir = result
        try:
            file_paths = json.loads(file_paths_json) if file_paths_json else {}
        except json.JSONDecodeError:
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
            color_palette = [
                'red', 'blue', 'green', 'orange', 'purple', 'cyan',
                'magenta', 'yellow', 'brown', 'pink', 'gray', 'olive',
                'navy', 'lime', 'maroon', 'teal', 'silver', 'gold'
            ]
            plot.colors = {}
            try:
                for i, material in enumerate(materials):
                    material_id = int(material.id)
                    if material_id > 0:
                        plot.colors[material_id] = color_palette[i % len(color_palette)]
            except Exception as color_error:
                logger.warning(f"Error setting preview colors: {color_error}")
            plots = openmc.Plots([plot])
            plots.export_to_xml()
            openmc.plot_geometry()
            preview_path = 'geometry_preview.png'
            if os.path.exists(preview_path):
                with open(preview_path, 'rb') as f:
                    preview_data = base64.b64encode(f.read()).decode()
                return jsonify({'success': True, 'preview': preview_data, 'message': 'Adaptive geometry preview generated successfully', 'analysis': {'geometry_type': geometry_type, 'dimensions': f'{width:.1f} x {height:.1f}', 'materials': num_materials, 'resolution': f'{plot.pixels[0]}x{plot.pixels[1]}'}})
            else:
                return jsonify({'error': 'Failed to generate geometry preview'}), 500
        finally:
            os.chdir(original_dir)
    except Exception as e:
        logger.error(f"Error generating geometry preview: {str(e)}")
        return jsonify({'error': f'Failed to generate preview: {str(e)}'}), 500


@app.route('/debug/<job_id>')
def debug_job(job_id):
    try:
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        result = c.fetchone()
        conn.close()
        if not result:
            return jsonify({'error': 'Job not found'}), 404
        conn = sqlite3.connect('jobs.db')
        c = conn.cursor()
        c.execute("PRAGMA table_info(jobs)")
        columns = [column[1] for column in c.fetchall()]
        conn.close()
        job_info = dict(zip(columns, result))
        job_dir = job_info.get('job_dir', f'jobs/{job_id}')
        file_system_info = {'job_dir_exists': os.path.exists(job_dir), 'job_dir_path': job_dir, 'files_in_dir': []}
        if os.path.exists(job_dir):
            try:
                files = os.listdir(job_dir)
                for file in files:
                    file_path = os.path.join(job_dir, file)
                    file_system_info['files_in_dir'].append({'name': file, 'size': os.path.getsize(file_path), 'exists': os.path.exists(file_path)})
            except Exception as e:
                file_system_info['error'] = str(e)
        file_paths_info = {}
        if job_info.get('file_paths'):
            try:
                file_paths = json.loads(job_info['file_paths'])
                if isinstance(file_paths, dict) and 'file_paths' in file_paths:
                    file_paths = file_paths['file_paths']
                for key, path in file_paths.items():
                    file_paths_info[key] = {'path': path, 'exists': os.path.exists(path), 'size': os.path.getsize(path) if os.path.exists(path) else 0}
            except Exception as e:
                file_paths_info['error'] = str(e)
        return jsonify({'job_info': job_info, 'file_system': file_system_info, 'file_paths': file_paths_info})
    except Exception as e:
        logger.error(f"Error debugging job {job_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    missing_libs = []
    for lib_name, lib_path in CROSS_SECTIONS_LIBRARIES.items():
        if not os.path.exists(lib_path):
            missing_libs.append(f"{lib_name}: {lib_path}")
    if missing_libs:
        print("WARNING: The following cross-sections files were not found:")
        for lib in missing_libs:
            print(f"  - {lib}")
        print("Please update CROSS_SECTIONS_LIBRARIES in app.py or ensure the files exist")
        print("The application will still start, but simulations using missing libraries will fail")
    else:
        print("All cross-sections libraries found successfully!")

    os.makedirs('jobs', exist_ok=True)
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)

