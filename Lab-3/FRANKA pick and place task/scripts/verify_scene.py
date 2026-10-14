import os
import sys
import mujoco

print("=== SEARCHING XML ASSETS ===")
for root, _, files in os.walk("robot_descriptions"):
    for f in files:
        if f.endswith(".xml"):
            path = os.path.join(root, f)
            print(f"Found file: {path}")
            with open(path, "r") as fp:
                for line in fp:
                    if "<include" in line:
                        print(f"   └─ Contains include: {line.strip()}")

scene_xml = "pick_place_scene.xml"
scene_dir = os.path.abspath("robot_descriptions")
full_scene_path = os.path.join(scene_dir, scene_xml)

print(f"\n=== ATTEMPTING LOAD FROM {scene_dir} ===")

if not os.path.exists(full_scene_path):
    print(f"[ERROR] Scene file not found at {full_scene_path}")
    sys.exit(1)

# Change directory context so MuJoCo resolves relative includes
original_dir = os.getcwd()
try:
    os.chdir(scene_dir)
    model = mujoco.MjModel.from_xml_path(scene_xml)
    data = mujoco.MjData(model)
    os.chdir(original_dir)

    print("\nSUCCESS! MuJoCo Model Loaded Successfully!")
    print(f"Total DOFs (nv): {model.nv} | Actuators (nu): {model.nu}")

    print("\n--- SITES ---")
    for i in range(model.nsite):
        print(f"  [{i}] '{model.site(i).name}'")

    print("\n--- BODIES ---")
    for i in range(model.nbody):
        print(f"  [{i}] '{model.body(i).name}'")

    print("\n--- JOINTS ---")
    for i in range(model.njnt):
        print(f"  [{i}] '{model.joint(i).name}'")

except Exception as e:
    os.chdir(original_dir)
    print(f"\n[ERROR] Loading failed: {e}")
    sys.exit(1)
