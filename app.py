from flask import Flask, request, jsonify
from flask_cors import CORS
import math
import random
import requests
import os

app = Flask(__name__)
CORS(app)

OSRM_URL = os.environ.get("OSRM_URL", None)
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
    """Single OSRM table API call - returns matrix in minutes."""
    n = len(points)
    try:
        coords = ";".join(f"{p['lng']},{p['lat']}" for p in points)
        url = f"{osrm_url}/table/v1/driving/{coords}?annotations=duration"
        r = requests.get(url, timeout=120)
        data = r.json()
        if data.get("code") != "Ok":
            raise ValueError(f"OSRM error: {data.get('code')}")
        matrix = [
            [v / 60 if v is not None else dist_km(points[i], points[j]) * 3
             for j, v in enumerate(row)]
            for i, row in enumerate(data["durations"])
        ]
        return matrix, True
    except Exception:
        matrix = [[dist_km(points[i], points[j]) * 3 for j in range(n)] for i in range(n)]
        return matrix, False

def route_drive_mins(idxs, matrix):
    """Nearest-neighbour route through cluster using OSRM matrix."""
    if len(idxs) <= 1:
        return 0.0
    remaining = list(idxs)
    current = remaining[0]
    remaining.remove(current)
    total = 0.0
    while remaining:
        nxt = min(remaining, key=lambda j: matrix[current][j])
        total += matrix[current][nxt]
        current = nxt
        remaining.remove(current)
    return round(total, 1)

# ── Exact port of the HTML tool algorithm ────────────────────────────────────
# This is a faithful Python translation of geoFirstCluster() from the HTML tool.
# Geography first, hard value cap, strict improvement convergence.

def geo_first_cluster(props, K, cap_value, matrix=None):
    n = len(props)
    total_val = sum(p["value"] for p in props)
    target = total_val / K

    # Distance function: use OSRM matrix if available, else straight-line
    def d(i, j):
        if matrix:
            return matrix[i][j]
        return dist_km(props[i], props[j]) * 3

    unassigned = list(range(n))
    c_idxs = []
    c_vals = []
    c_cents = []

    def get_cen(idxs):
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
            dd = min(dist_km(props[i], c) for c in c_cents)
            if dd > bd:
                bd = dd
                best = i
        return best

    # Phase 1: geography-first greedy build
    # Seed from most isolated point, expand by nearest neighbour
    # Stop at target value. Cap is hard ceiling only.
    while len(c_idxs) < K and unassigned:
        si = find_seed()
        if si < 0:
            break
        idxs = [si]
        val = props[si]["value"]
        unassigned.remove(si)
        c_cents.append(get_cen(idxs))

        while unassigned:
            if val >= target:
                break

            cen = c_cents[-1]
            sample = unassigned if len(unassigned) <= 1000 else random.sample(unassigned, 1000)

            # Find nearest unassigned — pure geography (straight-line to centroid)
            # This matches the HTML tool exactly
            bi, bd = -1, float("inf")
            for idx in sample:
                if props[idx]["value"] > cap_value:
                    continue
                dd = dist_km(props[idx], cen)
                if dd < bd:
                    bd = dd
                    bi = idx

            if bi < 0:
                break
            if val + props[bi]["value"] > cap_value:
                break

            idxs.append(bi)
            val += props[bi]["value"]
            unassigned.remove(bi)
            c_cents[-1] = get_cen(idxs)

        c_idxs.append(idxs)
        c_vals.append(val)

    # Assign stragglers to nearest cluster
    for i in unassigned:
        bk = min(range(len(c_cents)), key=lambda k: dist_km(props[i], c_cents[k]))
        c_idxs[bk].append(i)
        c_vals[bk] += props[i]["value"]
        c_cents[bk] = get_cen(c_idxs[bk])

    # Phase 2: strict improvement convergence
    # Each job moves only if it is strictly closer to another cluster centroid
    # AND that cluster stays under cap. No oscillation possible.
    # Uses straight-line to centroids for speed (matches HTML tool)
    K_NEARBY = 12
    for _ in range(500):
        cents = [get_cen(idxs) for idxs in c_idxs]

        # Build nearby lookup: each cluster checks only its 12 nearest neighbours
        nearby = []
        for k in range(len(c_idxs)):
            dists = sorted(
                range(len(c_idxs)),
                key=lambda j: dist_km(cents[k], cents[j]) if j != k else float("inf")
            )
            nearby.append(dists[:K_NEARBY])

        moves = 0
        for k in range(len(c_idxs)):
            for i in list(c_idxs[k]):
                cur_d = dist_km(props[i], cents[k])
                best_j, best_d = -1, cur_d

                for j in nearby[k]:
                    if c_vals[j] + props[i]["value"] > cap_value:
                        continue
                    dd = dist_km(props[i], cents[j])
                    if dd < best_d:
                        best_d = dd
                        best_j = j

                if best_j >= 0:
                    c_idxs[k].remove(i)
                    c_idxs[best_j].append(i)
                    c_vals[k] -= props[i]["value"]
                    c_vals[best_j] += props[i]["value"]
                    cents[k] = get_cen(c_idxs[k] if c_idxs[k] else [0])
                    cents[best_j] = get_cen(c_idxs[best_j])
                    moves += 1

        if moves == 0:
            break

    return c_idxs, c_vals, c_cents

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
        use_osrm = data.get("use_osrm", True)

        dataset_id = data.get("dataset_id", None)
        if dataset_id and dataset_id in PROPERTY_STORE:
            props = PROPERTY_STORE[dataset_id]
        else:
            props = data.get("properties", [])
            props = [p for p in props if float(p.get("value", 0)) > 0.05]

        if not props:
            return jsonify({"error": "No properties found. Upload first or include properties in request."}), 400

        # Get OSRM matrix once for route distance calculation
        matrix, osrm_used = None, False
        if osrm_url and use_osrm:
            matrix, osrm_used = get_osrm_table(props, osrm_url)

        # Run clustering (exact HTML algorithm)
        c_idxs, c_vals, c_cents = geo_first_cluster(props, K, cap_value, matrix=None)

        # Build result — use OSRM matrix for route distances if available
        clusters = []
        for k, idxs in enumerate(c_idxs):
            if not idxs:
                continue
            pts = [props[i] for i in idxs]
            cen = c_cents[k] if k < len(c_cents) else centroid(pts)
            rdm = route_drive_mins(idxs, matrix) if matrix else 0
            clusters.append({
                "cluster_id": k + 1,
                "properties": pts,
                "value": round(c_vals[k], 2),
                "job_count": len(pts),
                "centroid_lat": round(cen["lat"], 6),
                "centroid_lng": round(cen["lng"], 6),
                "route_drive_mins": rdm
            })

        clusters.sort(key=lambda c: -c["value"])
        total_val = sum(p["value"] for p in props)
        target = round(total_val / K, 2)

        return jsonify({
            "clusters": clusters,
            "target_value": target,
            "cluster_count": len(clusters),
            "total_properties": sum(c["job_count"] for c in clusters),
            "osrm_used": osrm_used
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

        results = []
        for cluster in clusters:
            pts = cluster["properties"]
            if not pts or cluster["value"] + new_prop["value"] > cap_value:
                continue

            all_pts = [new_prop] + pts
            if osrm_url:
                mat, _ = get_osrm_table(all_pts, osrm_url)
                def t(i, j): return mat[i][j]
            else:
                def t(i, j): return dist_km(all_pts[i], all_pts[j]) * 3

            cen = {"lat": cluster["centroid_lat"], "lng": cluster["centroid_lng"]}
            remaining = list(range(1, len(all_pts)))
            ordered = []
            cur = min(remaining, key=lambda i: dist_km(all_pts[i], cen))
            remaining.remove(cur)
            ordered.append(cur)
            while remaining:
                nxt = min(remaining, key=lambda j: t(ordered[-1], j))
                ordered.append(nxt)
                remaining.remove(nxt)

            best_cost, best_pos = float("inf"), 0
            cost = t(0, ordered[0])
            if cost < best_cost: best_cost = cost; best_pos = 0
            for pos in range(len(ordered) - 1):
                a, b = ordered[pos], ordered[pos + 1]
                ins = t(a, 0) + t(0, b) - t(a, b)
                if ins < best_cost: best_cost = ins; best_pos = pos + 1
            cost = t(ordered[-1], 0)
            if cost < best_cost: best_cost = cost; best_pos = len(ordered)

            nearest = pts[ordered[0] - 1]
            results.append({
                "cluster_id": cluster["cluster_id"],
                "cluster_value": cluster["value"],
                "cluster_jobs": cluster["job_count"],
                "headroom": round(cap_value - cluster["value"], 2),
                "insertion_cost_mins": round(max(0, best_cost), 1),
                "nearest_stop_address": nearest.get("address", ""),
                "nearest_stop_dist_km": round(dist_km(new_prop, nearest), 2),
                "centroid_lat": cluster["centroid_lat"],
                "centroid_lng": cluster["centroid_lng"],
                "score": round(best_cost + dist_km(new_prop, nearest) * 2, 2)
            })

        results.sort(key=lambda r: r["score"])
        return jsonify({"new_property": new_prop, "options": results[:5]})

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
        matrix, _ = get_osrm_table(props, osrm_url) if osrm_url else (None, False)
        c_idxs, c_vals, c_cents = geo_first_cluster(props, K, cap_value, matrix=None)

        clusters = []
        for k, idxs in enumerate(c_idxs):
            if not idxs: continue
            pts = [props[i] for i in idxs]
            cen = c_cents[k] if k < len(c_cents) else centroid(pts)
            rdm = route_drive_mins(idxs, matrix) if matrix else 0
            clusters.append({
                "cluster_id": k + 1,
                "properties": pts,
                "value": round(c_vals[k], 2),
                "job_count": len(pts),
                "centroid_lat": round(cen["lat"], 6),
                "centroid_lng": round(cen["lng"], 6),
                "route_drive_mins": rdm
            })

        clusters.sort(key=lambda c: -c["value"])
        total_val = sum(p["value"] for p in props)
        return jsonify({
            "clusters": clusters,
            "target_value": round(total_val / K, 2),
            "cluster_count": len(clusters)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
