import os
import sys
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Get the directory where this app.py file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the src directory to Python path for importing stanlogic
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

try:
    from stanlogic import KMapSolver
    from stanlogic.BoolMinGeo import KMapSolver3D
except ImportError as e:
    print(f"Error: Could not import stanlogic modules: {e}")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

# Flask app with BASE_DIR as static folder
app = Flask(__name__, static_folder=BASE_DIR, static_url_path='')
CORS(app)

@app.route('/')
def index():
    """Serves the main K-Map solver visualizer page."""
    return send_from_directory(BASE_DIR, 'kmap-solver-visualizer.html')

@app.route('/images/<path:filename>')
def serve_images(filename):
    """Serves image files."""
    return send_from_directory(os.path.join(BASE_DIR, 'images'), filename)

@app.route('/<path:filename>')
def static_files(filename):
    """Serves other static files."""
    return send_from_directory(BASE_DIR, filename)

@app.route('/api/solve', methods=['POST'])
def solve_kmap():
    """
    API endpoint to solve a K-map and return visualization steps.
    Accepts a JSON payload with either:
    - 2D mode: 'kmap', 'convention', 'form'
    - 3D mode: 'num_vars', 'output_values', 'convention', 'form', 'is3D'
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid input: No data provided.'}), 400

        convention = data.get('convention', 'vranesic')
        form = data.get('form', 'sop')
        is3D = data.get('is3D', False)

        if is3D:
            # 3D K-Map mode (5+ variables)
            if 'num_vars' not in data or 'output_values' not in data:
                return jsonify({'success': False, 'error': 'Invalid input: num_vars and output_values required for 3D mode.'}), 400
            
            num_vars = data['num_vars']
            output_values = data['output_values']
            
            # Initialize 3D solver
            solver = KMapSolver3D(num_vars=num_vars, output_values=output_values)
            
            # Get minimized expression
            terms, expression = solver.minimize_3d(form=form)
            
            # For now, return simplified steps (visualization for 3D is complex)
            # You can extend this to provide detailed 3D visualization steps
            steps = {
                'allGroups': {'count': 0, 'coords': [], 'terms': []},
                'primeImplicants': {'count': len(terms), 'coords': [], 'terms': terms},
                'primeWithCoverage': {'coords': [], 'terms': terms, 'coverageCounts': []},
                'essentialPrimes': {'indices': [], 'coords': [], 'terms': []},
                'greedySelections': [],
                'finalSelected': {'coords': [], 'terms': terms}
            }
            
            return jsonify({
                'success': True,
                'expression': expression if expression else '0' if form == 'sop' else '1',
                'steps': steps,
                'targetValue': 0 if form == 'pos' else 1,
                'is3D': True,
                'num_vars': num_vars
            })
        else:
            # 2D K-Map mode (2-4 variables)
            if 'kmap' not in data:
                return jsonify({'success': False, 'error': 'Invalid input: kmap data missing.'}), 400

            kmap_data = data['kmap']
            
            # Initialize the solver with the provided data
            solver = KMapSolver(kmap=kmap_data, convention=convention)
            
            # Use the new visualization method to get detailed steps
            expression, steps = solver.minimize_visualize(form=form)

            # Return the successful result in the format expected by the frontend
            return jsonify({
                'success': True,
                'expression': expression,
                'steps': steps,
                'targetValue': 0 if form == 'pos' else 1,
                'is3D': False
            })

    except Exception as e:
        # Return a generic error message for any exceptions during solving
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Starting Flask server for K-Map Visualizer...")
    print(f"Base directory: {BASE_DIR}")
    print(f"HTML file: {os.path.join(BASE_DIR, 'kmap-solver-visualizer.html')}")
    print(f"File exists: {os.path.exists(os.path.join(BASE_DIR, 'kmap-solver-visualizer.html'))}")
    print("=" * 60)
    print("View at: http://localhost:5000")
    print("Or from network: http://<your-ip>:5000")
    print("=" * 60)
    # Set debug=True for development
    app.run(host='0.0.0.0', port=5000, debug=True)
