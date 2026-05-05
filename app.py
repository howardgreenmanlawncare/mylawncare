from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import random
import requests
import os

app = Flask(__name__)
CORS(app)

# Read OSRM URL from environment variable — set this in Railway variables
OSRM_URL = os.environ.get('OSRM_URL', None)

# ── Helpers ───────────────────────────────────────────────────────────────────

def dist_km(a, b):
    dlat = (a['lat'] - b['lat']) * 111
    dlng = (a['lng'] - b['lng']) * 111 * math.cos(a['lat'] * math.pi / 180)
    return math.sqrt(dlat * dlat + dlng * dlng)

def get_osrm_duration(a, b, osrm_url):
    try:
        url = f"{osrm_url}/route/v1/driving/{a['lng']},{a['lat']};{b['lng']},{b['lat']}?overview=false"
        r = requests.get(url, timeout=5)
        data = r.json()
        return data['routes'][0]['duration'] / 60
    except Exception:
        return dist_km(a, b) * 3

def get_duration_matrix(points, osrm_url):
    try:
        coords = ';'.join(f"{p['lng']},{p['lat']}" for p in points)
        url = f"{osrm_url}/table/v1/driving/{coords}?annotations=duration"
        r = requests.get(url, timeout=30)
        data = r.json()
        matrix = [[v / 60 for v in row] for row in data['durations']]
        return matrix
    except Exception:
        n = len(points)
        return [[dist_km(points[i], points[j]) * 3 for j in range(n)] for i in range(n)]

def centroid(pts):
    if not pts:
        return {'lat': 0, 'lng': 0}
    return {
        'lat': sum(p['lat'] for p in pts) / len(pts),
        'lng': sum(p['lng'] for p in pts) / len(pts)
    }

def nearest_neighbour_route_km(pts, osrm_url=None):
    if len(pts) <= 1:
        return 0.0
    cen = centroid(pts)
    remaining = list(pts)
    current = cen
    total = 0.0
    ordered = []
    while remaining:
        nearest = min(remaining, key=lambda p: dist_km(p, current))
        if ordered:
            total += dist_km(ordered[-1], nearest)
        ordered.append(nearest)
        current = nearest
        remaining.remove(nearest)
    return total

# ── Clustering ────────────────────────────────────────────────────────────────

def geo_first_cluster(props, K, cap_value, osrm_url=None):
    n = len(props)
    total_val = sum(p['value'] for p in props)
    target = total_val / K

    unassigned = list(range(n))
    c_idxs = []
    c_vals = []
    c_cents = []

    def get_cen(idxs):
        if not idxs:
            return {'lat': 0, 'lng': 0}
        return {
            'lat': sum(props[i]['lat'] for i in idxs) / len(idxs),
            'lng': sum(props[i]['lng'] for i in idxs) / len(idxs)
        }

    def find_seed():
        if not c_cents:
            return unassigned[random.randint(0, len(unassigned) - 1)]
        best, bd = -1, -1
        sample = unassigned if len(unassigned) <= 400 else random.sample(unassigned, 400)
        for i in sample:
            d = min(dist_km(props[i], c) for c in c_cents)
            if d > bd:
                bd = d
                best = i
        return best

    # Phase 1: greedy geographic build — stop at target, cap is hard ceiling only
    while len(c_idxs) < K and unassigned:
        si = find_seed()
        if si < 0:
            break
        idxs = [si]
        val = props[si]['value']
        unassigned.remove(si)
        c_cents.append(get_cen(idxs))

        while unassigned:
            if val >= target:
                break
            cen = c_cents[-1]
            sample = unassigned if len(unassigned) <= 1000 else random.sample(unassigned, 1000)
            bi, bd = -1, float('inf')
            for idx in sample:
                if props[idx]['value'] > cap_value:
                    continue
                d = dist_km(props[idx], cen)
                if d < bd:
                    bd = d
                    bi = idx
            if bi < 0:
                break
            if val + props[bi]['value'] > cap_value:
                break
            idxs.append(bi)
            val += props[bi]['value']
            unassigned.remove(bi)
            c_cents[-1] = get_cen(idxs)

        c_idxs.append(idxs)
        c_vals.append(val)

    # Assign stragglers
    for i in unassigned:
        bk = min(range(len(c_cents)), key=lambda k: dist_km(props[i], c_cents[k]))
        c_idxs[bk].append(i)
        c_vals[bk] += props[i]['value']
        c_cents[bk] = get_cen(c_idxs[bk])

    # Phase 2: strict improvement convergence using OSRM if available
    K_NEARBY = 12
    for _ in range(200):
        cents = [get_cen(idxs) for idxs in c_idxs]
        nearby = []
        for k in range(len(c_idxs)):
            dists = sorted(range(len(c_idxs)),
                           key=lambda j: dist_km(cents[k], cents[j]) if j != k else float('inf'))
            nearby.append(dists[:K_NEARBY])

        moves = 0
        for k in range(len(c_idxs)):
            for i in list(c_idxs[k]):
                cur_d = dist_km(props[i], cents[k])
                best_j, best_d = -1, cur_d
                for j in nearby[k]:
                    if c_vals[j] + props[i]['value'] > cap_value:
                        continue
                    d = dist_km(props[i], cents[j])
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    c_idxs[k].remove(i)
                    c_idxs[best_j].append(i)
                    c_vals[k] -= props[i]['value']
                    c_vals[best_j] += props[i]['value']
                    cents[k] = get_cen(c_idxs[k] if c_idxs[k] else [i])
                    cents[best_j] = get_cen(c_idxs[best_j])
                    moves += 1
        if moves == 0:
            break

    # Build result
    clusters = []
    for k, idxs in enumerate(c_idxs):
        if not idxs:
            continue
        pts = [props[i] for i in idxs]
        cen = get_cen(idxs)
        route_km = nearest_neighbour_route_km(pts, osrm_url)
        clusters.append({
            'cluster_id': k + 1,
            'properties': pts,
            'value': round(c_vals[k], 2),
            'job_count': len(pts),
            'centroid_lat': round(cen['lat'], 6),
            'centroid_lng': round(cen['lng'], 6),
            'route_km': round(route_km, 2)
        })

    clusters.sort(key=lambda c: -c['value'])
    return clusters, round(target, 2)

# ── En-route finder ───────────────────────────────────────────────────────────

def find_best_insertion(new_prop, clusters, cap_value, osrm_url=None):
    results = []

    for cluster in clusters:
        pts = cluster['properties']
        if not pts:
            continue
        if cluster['value'] + new_prop['value'] > cap_value:
            continue

        cen = {'lat': cluster['centroid_lat'], 'lng': cluster['centroid_lng']}
        remaining = list(pts)
        ordered = []
        current = cen
        while remaining:
            nearest = min(remaining, key=lambda p: dist_km(p, current))
            ordered.append(nearest)
            current = nearest
            remaining.remove(nearest)

        best_cost = float('inf')
        best_position = 0

        if len(ordered) == 0:
            best_cost = dist_km(new_prop, cen) * 3
        elif len(ordered) == 1:
            best_cost = dist_km(new_prop, ordered[0]) * 3 * 2
        else:
            cost_start = dist_km(new_prop, ordered[0]) * 3
            if cost_start < best_cost:
                best_cost = cost_start
                best_position = 0

            for i in range(len(ordered) - 1):
                a, b = ordered[i], ordered[i + 1]
                direct = dist_km(a, b) * 3
                via_new = dist_km(a, new_prop) * 3 + dist_km(new_prop, b) * 3
                insertion_cost = via_new - direct
                if insertion_cost < best_cost:
                    best_cost = insertion_cost
                    best_position = i + 1

            cost_end = dist_km(ordered[-1], new_prop) * 3
            if cost_end < best_cost:
                best_cost = cost_end
                best_position = len(ordered)

        # Use real road time if OSRM available
        if osrm_url:
            try:
                if best_position == 0:
                    a, b = new_prop, ordered[0]
                elif best_position >= len(ordered):
                    a, b = ordered[-1], new_prop
                else:
                    a = ordered[best_position - 1]
                    b = ordered[best_position]
                direct = get_osrm_duration(a, b, osrm_url)
                via = (get_osrm_duration(a, new_prop, osrm_url) +
                       get_osrm_duration(new_prop, b, osrm_url))
                best_cost = max(0, via - direct)
            except Exception:
                pass

        nearest_stop = min(pts, key=lambda p: dist_km(p, new_prop))
        nearest_dist = round(dist_km(new_prop, nearest_stop), 2)

        results.append({
            'cluster_id': cluster['cluster_id'],
            'cluster_value': cluster['value'],
            'cluster_jobs': cluster['job_count'],
            'headroom': round(cap_value - cluster['value'], 2),
            'insertion_cost_mins': round(max(0, best_cost), 1),
            'insert_after_position': best_position,
            'nearest_stop_address': nearest_stop.get('address', ''),
            'nearest_stop_dist_km': nearest_dist,
            'centroid_lat': cluster['centroid_lat'],
            'centroid_lng': cluster['centroid_lng'],
            'score': round(best_cost + nearest_dist * 2, 2)
        })

    results.sort(key=lambda r: r['score'])
    return results[:5]

# ── In-memory property store ──────────────────────────────────────────────────
# Stores uploaded properties so /cluster doesn't need them in the request body
PROPERTY_STORE = {}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload():
    """
    POST /upload
    Body: { "properties": [...], "dataset_id": "my_dataset" }
    Stores properties server-side. Use dataset_id in /cluster calls.
    """
    try:
        data = request.get_json()
        props = data.get('properties', [])
        dataset_id = data.get('dataset_id', 'default')
        props = [p for p in props if float(p.get('value', 0)) > 0.05]
        PROPERTY_STORE[dataset_id] = props
        total_val = sum(p['value'] for p in props)
        return jsonify({
            'dataset_id': dataset_id,
            'property_count': len(props),
            'total_value': round(total_val, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    osrm_status = 'connected' if OSRM_URL else 'not configured'
    return jsonify({
        'status': 'ok',
        'service': 'mylawncare-api',
        'osrm': osrm_status,
        'osrm_url': OSRM_URL
    })

@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        data = request.get_json()
        K = int(data.get('crew_runs', 100))
        cap_value = float(data.get('cap_value', 700))
        osrm_url = data.get('osrm_url', OSRM_URL)

        # Support both inline properties and dataset_id reference
        dataset_id = data.get('dataset_id', None)
        if dataset_id and dataset_id in PROPERTY_STORE:
            props = PROPERTY_STORE[dataset_id]
        else:
            props = data.get('properties', [])
            props = [p for p in props if float(p.get('value', 0)) > 0.05]

        if not props:
            return jsonify({'error': 'No properties found. Upload first or include properties in request.'}), 400
        if K < 1:
            return jsonify({'error': 'crew_runs must be at least 1'}), 400
        clusters, target = geo_first_cluster(props, K, cap_value, osrm_url)

        return jsonify({
            'clusters': clusters,
            'target_value': target,
            'cluster_count': len(clusters),
            'total_properties': sum(c['job_count'] for c in clusters),
            'osrm_used': osrm_url is not None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/enroute', methods=['POST'])
def enroute():
    try:
        data = request.get_json()
        new_prop = data.get('new_property')
        clusters = data.get('clusters', [])
        cap_value = float(data.get('cap_value', 700))
        osrm_url = data.get('osrm_url', OSRM_URL)

        if not new_prop:
            return jsonify({'error': 'new_property is required'}), 400
        if not clusters:
            return jsonify({'error': 'clusters are required'}), 400

        options = find_best_insertion(new_prop, clusters, cap_value, osrm_url)

        return jsonify({
            'new_property': new_prop,
            'options': options,
            'osrm_used': osrm_url is not None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/rebalance', methods=['POST'])
def rebalance():
    try:
        data = request.get_json()
        props = data.get('properties', [])
        K = int(data.get('crew_runs', 1))
        cap_value = float(data.get('cap_value', 700))
        osrm_url = data.get('osrm_url', OSRM_URL)

        if not props:
            return jsonify({'error': 'No properties provided'}), 400

        props = [p for p in props if float(p.get('value', 0)) > 0.05]
        clusters, target = geo_first_cluster(props, K, cap_value, osrm_url)

        return jsonify({
            'clusters': clusters,
            'target_value': target,
            'cluster_count': len(clusters),
            'osrm_used': osrm_url is not None
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
