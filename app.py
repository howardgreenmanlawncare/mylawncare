from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import random
import requests
import os

app = Flask(__name__)
CORS(app)

OSRM_URL = os.environ.get('OSRM_URL', None)

# ── In-memory property store ───────────────────────────────────────────────────
PROPERTY_STORE = {}

# ── Helpers ───────────────────────────────────────────────────────────────────

def dist_km(a, b):
    dlat = (a["lat"] - b["lat"]) * 111
    dlng = (a["lng"] - b["lng"]) * 111 * math.cos(a["lat"] * math.pi / 180)
    return math.sqrt(dlat * dlat + dlng * dlng)

def centroid(pts):
    if not pts:
        return {"lat": 0, "lng": 0}
    return {
        "lat": sum(p["lat"] for p in pts) / len(pts),
        "lng": sum(p["lng"] for p in pts) / len(pts)
    }

def get_osrm_table(points, osrm_url):
    """
    Fetch full travel time matrix from OSRM table API in ONE call.
    Returns matrix[i][j] = travel time in minutes from point i to point j.
    Falls back to straight-line if OSRM unavailable.
    """
    n = len(points)
    try:
        coords = ";".join(f"{p['lng']},{p['lat']}" for p in points)
        url = f"{osrm_url}/table/v1/driving/{coords}?annotations=duration"
        r = requests.get(url, timeout=60)
        data = r.json()
        if data.get("code") != "Ok":
            raise ValueError(f"OSRM error: {data.get('code')}")
        # Convert seconds to minutes
        matrix = [[v / 60 if v is not None else dist_km(points[i], points[j]) * 3
                   for j, v in enumerate(row)]
                  for i, row in enumerate(data["durations"])]
        return matrix, True
    except Exception as e:
        # Fallback: straight-line * 3 mins/km
        matrix = [[dist_km(points[i], points[j]) * 3 for j in range(n)] for i in range(n)]
        return matrix, False

def nearest_neighbour_route(pts, matrix, idx_map):
    """
    Calculate route distance using nearest-neighbour heuristic with real matrix.
    idx_map maps local pt index to matrix index.
    """
    if len(pts) <= 1:
        return 0.0
    cen = centroid(pts)
    remaining = list(range(len(pts)))
    # Start from point nearest to centroid
    current = min(remaining, key=lambda i: dist_km(pts[i], cen))
    remaining.remove(current)
    total = 0.0
    while remaining:
        mi = idx_map[current]
        nearest = min(remaining, key=lambda j: matrix[mi][idx_map[j]])
        total += matrix[mi][idx_map[nearest]]
        current = nearest
        remaining.remove(nearest)
    return round(total, 1)

# ── Clustering ────────────────────────────────────────────────────────────────

def geo_first_cluster(props, K, cap_value, osrm_url=None):
    n = len(props)
    total_val = sum(p["value"] for p in props)
    target = total_val / K

    # ── Fetch full OSRM matrix in ONE call ──
    matrix, osrm_used = get_osrm_table(props, osrm_url) if osrm_url else (
        [[dist_km(props[i], props[j]) * 3 for j in range(n)] for i in range(n)], False
    )

    def travel(i, j):
        return matrix[i][j]

    unassigned = list(range(n))
    c_idxs = []
    c_vals = []
    c_cents = []

    def get_cen_idx(idxs):
        if not idxs:
            return {"lat": 0, "lng": 0}
        return {
            "lat": sum(props[i]["lat"] for i in idxs) / len(idxs),
            "lng": sum(props[i]["lng"] for i in idxs) / len(idxs)
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

    # Phase 1: greedy geographic build using OSRM travel times
    while len(c_idxs) < K and unassigned:
        si = find_seed()
        if si < 0:
            break
        idxs = [si]
        val = props[si]["value"]
        unassigned.remove(si)
        c_cents.append(get_cen_idx(idxs))

        while unassigned:
            if val >= target:
                break
            # Find nearest unassigned using OSRM travel time from centroid
            # Use nearest property in cluster as proxy for centroid travel time
            cen_idx = idxs[0]  # use first prop as anchor
            best_t = float("inf")
            bi = -1
            sample = unassigned if len(unassigned) <= 1000 else random.sample(unassigned, 1000)
            for idx in sample:
                if props[idx]["value"] > cap_value:
                    continue
                # Use min travel time from any cluster member to candidate
                t = min(travel(i, idx) for i in idxs[-3:])  # check last 3 for speed
                if t < best_t:
                    best_t = t
                    bi = idx
            if bi < 0:
                break
            if val + props[bi]["value"] > cap_value:
                break
            idxs.append(bi)
            val += props[bi]["value"]
            unassigned.remove(bi)
            c_cents[-1] = get_cen_idx(idxs)

        c_idxs.append(idxs)
        c_vals.append(val)

    # Assign stragglers
    for i in unassigned:
        bk = min(range(len(c_cents)), key=lambda k: dist_km(props[i], c_cents[k]))
        c_idxs[bk].append(i)
        c_vals[bk] += props[i]["value"]
        c_cents[bk] = get_cen_idx(c_idxs[bk])

    # Phase 2: convergence using OSRM matrix
    K_NEARBY = 12
    for _ in range(200):
        cents = [get_cen_idx(idxs) for idxs in c_idxs]
        nearby = []
        for k in range(len(c_idxs)):
            dists = sorted(range(len(c_idxs)),
                           key=lambda j: dist_km(cents[k], cents[j]) if j != k else float("inf"))
            nearby.append(dists[:K_NEARBY])

        moves = 0
        for k in range(len(c_idxs)):
            for i in list(c_idxs[k]):
                # Use avg travel time from i to all members of cluster k
                cur_d = sum(travel(i, j) for j in c_idxs[k] if j != i) / max(1, len(c_idxs[k]) - 1) if len(c_idxs[k]) > 1 else 0
                best_j, best_d = -1, cur_d
                for j in nearby[k]:
                    if c_vals[j] + props[i]["value"] > cap_value:
                        continue
                    # Travel time from i to cluster j centroid (approx via nearest member)
                    if not c_idxs[j]:
                        continue
                    d = min(travel(i, m) for m in c_idxs[j][:5])
                    if d < best_d:
                        best_d = d
                        best_j = j
                if best_j >= 0:
                    c_idxs[k].remove(i)
                    c_idxs[best_j].append(i)
                    c_vals[k] -= props[i]["value"]
                    c_vals[best_j] += props[i]["value"]
                    cents[k] = get_cen_idx(c_idxs[k] if c_idxs[k] else [props[i]])
                    cents[best_j] = get_cen_idx(c_idxs[best_j])
                    moves += 1
        if moves == 0:
            break

    # Build result with real road route distances
    clusters = []
    for k, idxs in enumerate(c_idxs):
        if not idxs:
            continue
        pts = [props[i] for i in idxs]
        cen = get_cen_idx(idxs)
        idx_map = {local: global_i for local, global_i in enumerate(idxs)}
        route_mins = nearest_neighbour_route(pts, matrix, idx_map)
        clusters.append({
            "cluster_id": k + 1,
            "properties": pts,
            "value": round(c_vals[k], 2),
            "job_count": len(pts),
            "centroid_lat": round(cen["lat"], 6),
            "centroid_lng": round(cen["lng"], 6),
            "route_drive_mins": route_mins,
            "osrm_used": osrm_used
        })

    clusters.sort(key=lambda c: -c["value"])
    return clusters, round(target, 2)

# ── En-route finder ───────────────────────────────────────────────────────────

def find_best_insertion(new_prop, clusters, cap_value, osrm_url=None):
    results = []
    for cluster in clusters:
        pts = cluster["properties"]
        if not pts or cluster["value"] + new_prop["value"] > cap_value:
            continue

        # Get travel times from new_prop to all stops in this cluster
        all_pts = [new_prop] + pts
        if osrm_url:
            mat, _ = get_osrm_table(all_pts, osrm_url)
            def t(i, j):
                return mat[i][j]
        else:
            def t(i, j):
                return dist_km(all_pts[i], all_pts[j]) * 3

        # Order stops by nearest neighbour
        cen = {"lat": cluster["centroid_lat"], "lng": cluster["centroid_lng"]}
        remaining = list(range(1, len(all_pts)))
        ordered_idxs = []
        current = min(remaining, key=lambda i: dist_km(all_pts[i], cen))
        remaining.remove(current)
        ordered_idxs.append(current)
        while remaining:
            nxt = min(remaining, key=lambda j: t(ordered_idxs[-1], j))
            ordered_idxs.append(nxt)
            remaining.remove(nxt)

        # Find cheapest insertion of new_prop (index 0) into ordered route
        best_cost = float("inf")
        best_pos = 0

        # Insert at start
        cost = t(0, ordered_idxs[0])
        if cost < best_cost:
            best_cost = cost
            best_pos = 0

        # Insert between consecutive stops
        for pos in range(len(ordered_idxs) - 1):
            a, b = ordered_idxs[pos], ordered_idxs[pos + 1]
            direct = t(a, b)
            via = t(a, 0) + t(0, b)
            insertion_cost = via - direct
            if insertion_cost < best_cost:
                best_cost = insertion_cost
                best_pos = pos + 1

        # Insert at end
        cost = t(ordered_idxs[-1], 0)
        if cost < best_cost:
            best_cost = cost
            best_pos = len(ordered_idxs)

        nearest_stop = pts[ordered_idxs[0] - 1] if ordered_idxs else pts[0]
        nearest_dist = round(dist_km(new_prop, nearest_stop), 2)
        nearest_mins = round(t(0, ordered_idxs[0]), 1) if ordered_idxs else 0

        results.append({
            "cluster_id": cluster["cluster_id"],
            "cluster_value": cluster["value"],
            "cluster_jobs": cluster["job_count"],
            "headroom": round(cap_value - cluster["value"], 2),
            "insertion_cost_mins": round(max(0, best_cost), 1),
            "nearest_stop_address": nearest_stop.get("address", ""),
            "nearest_stop_dist_km": nearest_dist,
            "nearest_stop_mins": nearest_mins,
            "centroid_lat": cluster["centroid_lat"],
            "centroid_lng": cluster["centroid_lng"],
            "score": round(best_cost + nearest_dist * 2, 2)
        })

    results.sort(key=lambda r: r["score"])
    return results[:5]

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "mylawncare-api",
        "osrm": "connected" if OSRM_URL else "not configured",
        "osrm_url": OSRM_URL
    })

@app.route("/upload", methods=["POST"])
def upload():
    try:
        data = request.get_json()
        props = data.get("properties", [])
        dataset_id = data.get("dataset_id", "default")
        props = [p for p in props if float(p.get("value", 0)) > 0.05]
        PROPERTY_STORE[dataset_id] = props
        total_val = sum(p["value"] for p in props)
        return jsonify({
            "dataset_id": dataset_id,
            "property_count": len(props),
            "total_value": round(total_val, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/cluster", methods=["POST"])
def cluster():
    try:
        data = request.get_json()
        K = int(data.get("crew_runs", 100))
        cap_value = float(data.get("cap_value", 700))
        osrm_url = data.get("osrm_url", OSRM_URL)

        dataset_id = data.get("dataset_id", None)
        if dataset_id and dataset_id in PROPERTY_STORE:
            props = PROPERTY_STORE[dataset_id]
        else:
            props = data.get("properties", [])
            props = [p for p in props if float(p.get("value", 0)) > 0.05]

        if not props:
            return jsonify({"error": "No properties found. Upload first or include properties in request."}), 400
        if K < 1:
            return jsonify({"error": "crew_runs must be at least 1"}), 400

        clusters, target = geo_first_cluster(props, K, cap_value, osrm_url)

        return jsonify({
            "clusters": clusters,
            "target_value": target,
            "cluster_count": len(clusters),
            "total_properties": sum(c["job_count"] for c in clusters),
            "osrm_used": clusters[0]["osrm_used"] if clusters else False
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/enroute", methods=["POST"])
def enroute():
    try:
        data = request.get_json()
        new_prop = data.get("new_property")
        clusters = data.get("clusters", [])
        cap_value = float(data.get("cap_value", 700))
        osrm_url = data.get("osrm_url", OSRM_URL)

        if not new_prop:
            return jsonify({"error": "new_property is required"}), 400
        if not clusters:
            return jsonify({"error": "clusters are required"}), 400

        options = find_best_insertion(new_prop, clusters, cap_value, osrm_url)
        return jsonify({"new_property": new_prop, "options": options})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/rebalance", methods=["POST"])
def rebalance():
    try:
        data = request.get_json()
        props = data.get("properties", [])
        K = int(data.get("crew_runs", 1))
        cap_value = float(data.get("cap_value", 700))
        osrm_url = data.get("osrm_url", OSRM_URL)

        if not props:
            return jsonify({"error": "No properties provided"}), 400

        props = [p for p in props if float(p.get("value", 0)) > 0.05]
        clusters, target = geo_first_cluster(props, K, cap_value, osrm_url)
        return jsonify({
            "clusters": clusters,
            "target_value": target,
            "cluster_count": len(clusters)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
