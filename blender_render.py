import bpy
import math
import os

# ---------------- CONFIG ----------------
OBJ_PATH = r"model\bed\IKEA_BEDDINGE\model.obj"
OUT_DIR = "renders_blender"
N_VIEWS = 24
ELEVATION_DEG = 20
IMG_SIZE = 512
# ----------------------------------------

# Reset scene
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import OBJ
bpy.ops.import_scene.obj(filepath=OBJ_PATH)
obj = bpy.context.selected_objects[0]

# Set render settings
bpy.context.scene.render.engine = "CYCLES"   # or "BLENDER_EEVEE"
bpy.context.scene.render.resolution_x = IMG_SIZE
bpy.context.scene.render.resolution_y = IMG_SIZE
bpy.context.scene.render.film_transparent = True  # so mask works

# Add camera
cam = bpy.data.objects.new("Camera", bpy.data.cameras.new("Camera"))
bpy.context.scene.collection.objects.link(cam)
bpy.context.scene.camera = cam

# Add light
light_data = bpy.data.lights.new(name="light", type="SUN")
light = bpy.data.objects.new(name="light", object_data=light_data)
bpy.context.collection.objects.link(light)
light.location = (5, -5, 5)

# Compute bounding box center + radius
center = 0.125 * sum((bpy.context.object.matrix_world @ v.co for v in obj.data.vertices), start=bpy.mathutils.Vector())
radius = max((obj.dimensions)) * 1.5

# Output folder
os.makedirs(OUT_DIR, exist_ok=True)

# Render loop
for i in range(N_VIEWS):
    angle = 2 * math.pi * i / N_VIEWS
    x = radius * math.cos(angle) * math.cos(math.radians(ELEVATION_DEG))
    y = radius * math.sin(angle) * math.cos(math.radians(ELEVATION_DEG))
    z = radius * math.sin(math.radians(ELEVATION_DEG))
    cam.location = (x, y, z)
    cam.rotation_euler = (math.radians(90 - ELEVATION_DEG), 0, angle + math.pi)

    # RGB render
    bpy.context.scene.render.filepath = os.path.join(OUT_DIR, f"rgb_{i:03d}.png")
    bpy.ops.render.render(write_still=True)

    # Mask render (object index)
    for o in bpy.data.objects:
        o.pass_index = 1
    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True

    # Enable compositor
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    for n in tree.nodes:
        tree.nodes.remove(n)

    rl = tree.nodes.new("CompositorNodeRLayers")
    id_mask = tree.nodes.new("CompositorNodeIDMask")
    id_mask.index = 1
    comp = tree.nodes.new("CompositorNodeComposite")

    tree.links.new(rl.outputs["IndexOB"], id_mask.inputs["ID value"])
    tree.links.new(id_mask.outputs["Alpha"], comp.inputs["Image"])

    bpy.context.scene.render.filepath = os.path.join(OUT_DIR, f"mask_{i:03d}.png")
    bpy.ops.render.render(write_still=True)

print("Done! Check", OUT_DIR)
