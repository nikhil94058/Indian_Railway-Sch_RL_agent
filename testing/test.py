import sys
import os
import time
import json
import heapq
from datetime import datetime, timedelta
from pathlib import Path

current_dir = Path(__file__).resolve().parent
root_dir = str(current_dir.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from my_env.client import MyEnv
from my_env.models import TrainAction
from my_env.inference import llm 
from my_env.data.network import data

# ==========================================
# 1. NETWORK & REGIONAL WEATHER DATA
# ==========================================
STATIONS = {s["id"]: {"name": s["name"], "platforms": s["platform_count"], "zone": s["zone"]} for s in data["stations"]}

GRAPH = {}
TRACKS = {} 
for t in data["tracks"]:
    u, v, w, c = t["from"], t["to"], t["travel_minutes"], t.get("track_count", 2)
    
    # Initialize the sub-dictionaries for GRAPH
    if u not in GRAPH: 
        GRAPH[u] = []
    if v not in GRAPH: 
        GRAPH[v] = []
        
    GRAPH[u].append((v, w))
    GRAPH[v].append((u, w))
    
    # Initialize the sub-dictionaries for TRACKS BEFORE adding to them!
    if u not in TRACKS: 
        TRACKS[u] = {}
    if v not in TRACKS: 
        TRACKS[v] = {}
        
    TRACKS[u][v] = {"time": w, "capacity": c}
    TRACKS[v][u] = {"time": w, "capacity": c}

ZONE_WEATHER = {
    "NR": {"condition": "Dense Fog", "multiplier": 1.5, "desc": "Northern fog causing 50% delay."},
    "NCR": {"condition": "Clear", "multiplier": 1.0, "desc": "Clear skies. Normal speeds."},
    "ECR": {"condition": "Heavy Rain", "multiplier": 1.2, "desc": "Monsoon rains causing 20% delay."},
    "ER": {"condition": "Clear", "multiplier": 1.0, "desc": "Clear skies. Normal speeds."},
    "NER": {"condition": "Clear", "multiplier": 1.0, "desc": "Clear skies. Normal speeds."},
    "SER": {"condition": "Cyclone Alert", "multiplier": 2.0, "desc": "Severe winds. Trains running at half speed."},
    "ECoR": {"condition": "Cyclone Alert", "multiplier": 2.0, "desc": "Severe winds. Trains running at half speed."}
}


# ==========================================
# 2. PATHFINDING & PHYSICS ALGORITHM
# ==========================================
def find_fastest_route(start: str, end: str) -> list:
    queue = [(0, start, [])] 
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited: continue
        visited.add(node)
        path = path + [node]
        if node == end: return path
        for neighbor, weight in GRAPH.get(node, []):
            if neighbor not in visited:
                heapq.heappush(queue, (cost + weight, neighbor, path))
    return []

def calculate_full_route(start: str, end: str, via: str = None) -> list:
    if not via: return find_fastest_route(start, end)
    leg1 = find_fastest_route(start, via)
    leg2 = find_fastest_route(via, end)
    if not leg1 or not leg2: return []
    return leg1[:-1] + leg2

def get_dynamic_travel_times(route, start_time, train_type):
    travel_data = []
    current_time = datetime.strptime(start_time, "%H:%M")
    day_offset = 0
    route_weather_log = []
    
    for i in range(len(route)):
        station_id = route[i]
        station_info = STATIONS.get(station_id, {})
        max_plats = station_info.get("platforms", 5)
        zone = station_info.get("zone", "NR")
        is_junction = station_info.get("is_junction", False)
        
        weather = ZONE_WEATHER.get(zone, {"condition": "Clear", "multiplier": 1.0})
        if f"{zone}: {weather['condition']}" not in route_weather_log:
            route_weather_log.append(f"{zone}: {weather['condition']}")
        
        if i == 0: 
            activity, min_dwell = "ORIGIN", 0
        elif i == len(route)-1: 
            activity, min_dwell = "DESTINATION", 0
        else:
            if train_type in ["Rajdhani", "Shatabdi", "Duronto"]:
                activity = "MAJOR_STOP" if is_junction else "THROUGH_PASS"
                min_dwell = 5 if activity == "MAJOR_STOP" else 1
            elif train_type == "Express":
                activity = "MAJOR_STOP" if max_plats >= 4 else "THROUGH_PASS"
                min_dwell = 10 if activity == "MAJOR_STOP" else 1
            elif train_type == "Passenger":
                activity, min_dwell = "MAJOR_STOP", 15
            else:
                activity = "MAJOR_STOP" if is_junction else "THROUGH_PASS"
                min_dwell = 30 if activity == "MAJOR_STOP" else 1

        if i > 0:
            prev_station = route[i-1]
            base_mins = TRACKS[prev_station][station_id]["time"]
            actual_mins = int(base_mins * weather["multiplier"]) 
            current_time += timedelta(minutes=actual_mins)
            if current_time.day > 1:
                day_offset += 1
                current_time = current_time.replace(day=1) 
        
        arr_str = current_time.strftime("%H:%M") if i > 0 else None
        
        if i < len(route)-1 and i > 0:
            current_time += timedelta(minutes=min_dwell)
            if current_time.day > 1:
                day_offset += 1
                current_time = current_time.replace(day=1)
                
        dep_str = current_time.strftime("%H:%M") if i < len(route)-1 else None
        next_track_cap = TRACKS[station_id][route[i+1]]["capacity"] if i < len(route)-1 else "N/A"

        # CHANGED KEYS: Clarify that the AI can delay departure
        travel_data.append({
            "station": station_id,
            "MAX_PLATFORMS_AVAILABLE": max_plats,
            "NEXT_TRACK_CAPACITY": next_track_cap,
            "activity": activity,
            "earliest_arrival": arr_str,
            "earliest_departure": dep_str,
            "day_offset": day_offset,
            "ASSIGN_PLATFORM_HERE": "integer"
        })
        
    return travel_data, " | ".join(route_weather_log)

# ==========================================
# 3. TASKS
# ==========================================
# The 15 Master Tasks designed to stress-test the AI
TASKS = [

# =========================================================
# 🔴 CORE CORRIDOR CHAOS (NDLS → DDU → HWH)
# Heavy congestion + overtakes
# =========================================================
{"id":"PASS-01","type":"Passenger","time":"04:50","src":"NDLS","dest":"CNB"},
{"id":"PASS-02","type":"Passenger","time":"05:10","src":"NDLS","dest":"PRYJ"},
{"id":"FR-03","type":"Freight","time":"05:20","src":"GZB","dest":"DDU"},
{"id":"EXP-04","type":"Express","time":"05:40","src":"NDLS","dest":"PNBE"},
{"id":"RAJ-05","type":"Rajdhani","time":"06:00","src":"NDLS","dest":"HWH"},
{"id":"SHAT-06","type":"Shatabdi","time":"06:05","src":"NDLS","dest":"LKO"},
{"id":"DUR-07","type":"Duronto","time":"06:20","src":"NDLS","dest":"PNBE"},
{"id":"FR-08","type":"Freight","time":"06:30","src":"ALJN","dest":"KIUL"},

# Overtake chain (fast trains catching slow)
{"id":"PASS-09","type":"Passenger","time":"06:45","src":"GZB","dest":"CNB"},
{"id":"RAJ-10","type":"Rajdhani","time":"07:00","src":"NDLS","dest":"BBS"},
{"id":"EXP-11","type":"Express","time":"07:10","src":"NDLS","dest":"DDU"},

# =========================================================
# 🔵 EASTERN SECTION (DDU → HWH → VSKP CYCLONE ZONE)
# Double travel time + queue buildup
# =========================================================
{"id":"EXP-12","type":"Express","time":"08:00","src":"HWH","dest":"VSKP"},
{"id":"FR-13","type":"Freight","time":"07:30","src":"ASN","dest":"BHC"},
{"id":"DUR-14","type":"Duronto","time":"08:30","src":"HWH","dest":"BBS"},
{"id":"PASS-15","type":"Passenger","time":"09:00","src":"SRC","dest":"KGP"},
{"id":"EXP-16","type":"Express","time":"09:15","src":"KGP","dest":"CTC"},
{"id":"FR-17","type":"Freight","time":"09:40","src":"KGP","dest":"VSKP"},
{"id":"SHAT-18","type":"Shatabdi","time":"10:00","src":"HWH","dest":"BBS"},

# =========================================================
# ⚫ HEAD-ON COLLISION GRID (2-track conflicts)
# =========================================================
{"id":"EXP-19","type":"Express","time":"13:30","src":"CNB","dest":"PNBE"},
{"id":"EXP-20","type":"Express","time":"13:35","src":"PNBE","dest":"CNB"},
{"id":"PASS-21","type":"Passenger","time":"13:40","src":"ARA","dest":"DDU"},
{"id":"FR-22","type":"Freight","time":"13:45","src":"DDU","dest":"ARA"},

# =========================================================
# 🟡 BOTTLENECK CLUSTER (DDU / MKA / KIUL overload)
# Platform saturation + forced dwell
# =========================================================
{"id":"PASS-23","type":"Passenger","time":"11:45","src":"BSB","dest":"MKA"},
{"id":"FR-24","type":"Freight","time":"11:30","src":"DDU","dest":"ASN"},
{"id":"SPL-25","type":"Special","time":"12:00","src":"PRYJ","dest":"JAJ"},
{"id":"EXP-26","type":"Express","time":"12:10","src":"MZP","dest":"KIUL"},
{"id":"RAJ-27","type":"Rajdhani","time":"12:20","src":"DDU","dest":"HWH"},
{"id":"PASS-28","type":"Passenger","time":"12:25","src":"BXR","dest":"PNBE"},
{"id":"FR-29","type":"Freight","time":"12:40","src":"PNBE","dest":"MKA"},

# =========================================================
# 🌙 LONG HAUL + MIDNIGHT CROSSING
# Multi-day cascading delays
# =========================================================
{"id":"EXP-30","type":"Express","time":"21:30","src":"FZR","dest":"HWH","via":"LKO"},
{"id":"RAJ-31","type":"Rajdhani","time":"22:00","src":"NDLS","dest":"BBS","via":"HWH"},
{"id":"FR-32","type":"Freight","time":"20:45","src":"CNB","dest":"VSKP","via":"KGP"},
{"id":"DUR-33","type":"Duronto","time":"23:00","src":"NDLS","dest":"HWH"},
{"id":"EXP-34","type":"Express","time":"23:30","src":"LKO","dest":"ASN"},

# =========================================================
# 🔁 REVERSE FLOW (DOWN TRAINS - HWH → NDLS)
# =========================================================
{"id":"RAJ-35","type":"Rajdhani","time":"06:30","src":"HWH","dest":"NDLS"},
{"id":"EXP-36","type":"Express","time":"07:00","src":"PNBE","dest":"NDLS"},
{"id":"PASS-37","type":"Passenger","time":"07:20","src":"DDU","dest":"CNB"},
{"id":"FR-38","type":"Freight","time":"07:45","src":"KIUL","dest":"GZB"},
{"id":"SHAT-39","type":"Shatabdi","time":"08:00","src":"LKO","dest":"NDLS"},
{"id":"EXP-40","type":"Express","time":"08:15","src":"PRYJ","dest":"NDLS"}

]
# ==========================================
# 4. RESILIENT AGENT LOOP
# ==========================================
def run_ml_agent():
    total_score = 0
    with MyEnv(base_url="http://localhost:8000").sync() as client:
        obs = client.reset().observation
        print(f"🚄 STARTING ENTERPRISE DISPATCHER EPISODE...\n")

        for step, task in enumerate(TASKS, 1):
            print(f"\n{'#'*70}\n🚆 TASK {step}: {task['id']} | {task['src']} -> {task['dest']} @ {task['time']}")
            
            calculated_route = calculate_full_route(task["src"], task["dest"])
            if not calculated_route: continue
                
            travel_data, weather_log = get_dynamic_travel_times(calculated_route, task["time"], task["type"])
            print(f"🌤️  WEATHER EN ROUTE: {weather_log}\n{'#'*70}")
            
            success = False
            attempts = 0
            max_attempts = 10
            last_error = "None"

            while not success and attempts < max_attempts:
                attempts += 1
                current_timetable = [t.model_dump() for t in obs.timetable]
                print(f"\n--- ATTEMPT {attempts}/{max_attempts} ---")
                
                prompt = f"""
[SYSTEM: RAILWAY DISPATCH ENGINE]
You are scheduling Train {task['id']} ({task['type']}). 

CRITICAL RULES FOR WAITING & DELAYS:
1. DYNAMIC WAITING: The SKELETON below provides the 'earliest_arrival' and 'earliest_departure'. 
   - If the PREVIOUS ERROR says the track or platform is congested, you MUST increase the departure time (e.g. wait 30 minutes at the station).
   - If you delay a departure, you MUST add that delay to all subsequent arrivals and departures in your output!
2. HEADWAY: Trains going in the SAME direction can follow each other if spaced 15 minutes apart.
3. PLATFORMS: Pick an integer between 1 and 'MAX_PLATFORMS_AVAILABLE'.

ROUTE SKELETON (These are Minimum Times! You can delay them!):
{json.dumps(travel_data, indent=2)}

CURRENT TIMETABLE: {current_timetable}
PREVIOUS ERROR TO FIX: {last_error}

Return ONLY raw JSON matching this format:
{{
  "action_type": "generate_schedule",
  "reasoning": "Explain if you delayed the train to wait for a track to clear.",
  "schedule_proposal": {{
    "train_id": "{task['id']}",
    "stops": [
       {{ "station": "NDLS", "arrival": null, "departure": "06:00", "platform": 2, "day_offset": 0 }}
    ]
  }}
}}
"""
                print("🧠 AI is analyzing congestion and calculating dynamic delays...")
                llm_response = llm(prompt_text=prompt)
                
                if isinstance(llm_response, str) or "John Doe" in str(llm_response):
                    print("⚠️ LLM Hallucinated. Retrying...")
                    continue

                try:
                    action = TrainAction(**llm_response)
                    result = client.step(action)
                    obs = result.observation
                    
                    if result.reward > 0:
                        print(f"🏆 SUCCESS! Reward: +{result.reward}")
                        print("📅 Final Route Scheduled:")
                        for s in action.schedule_proposal.stops:
                            print(f"   [{s.station: <4}] Plat: {s.platform} | Arr: {s.arrival or '--:--'} | Dep: {s.departure or '--:--'}")
                        print(f"🗣️ AI Reasoning: {action.reasoning}")
                        total_score += result.reward
                        success = True
                    else:
                        last_error = obs.server_feedback or "Unknown physics error."
                        print(f"❌ SERVER REJECTED: {last_error}")

                except Exception as e:
                    print(f"⚠️ JSON Validation Error: {e}")
                    continue

        print(f"\n🏁 EPISODE COMPLETE! Total Score: {total_score}")

if __name__ == "__main__":
    run_ml_agent()