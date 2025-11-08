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
except ImportError as e:
    print(f"Error: Could not import KMapSolver: {e}")
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
    Accepts a JSON payload with 'kmap', 'convention', and 'form'.
    """
    try:
        data = request.get_json()
        if not data or 'kmap' not in data:
            return jsonify({'success': False, 'error': 'Invalid input: K-map data missing.'}), 400

        kmap_data = data['kmap']
        convention = data.get('convention', 'vranesic')
        form = data.get('form', 'sop')

        # Initialize the solver with the provided data
        solver = KMapSolver(kmap=kmap_data, convention=convention)
        
        # Use the new visualization method to get detailed steps
        expression, steps = solver.minimize_visualize(form=form)

        # Return the successful result in the format expected by the frontend
        return jsonify({
            'success': True,
            'expression': expression,
            'steps': steps,
            'targetValue': 0 if form == 'pos' else 1
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
