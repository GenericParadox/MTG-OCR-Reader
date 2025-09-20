bl_info = {
    "name": "MTG OCR Trainer Tools",
    "blender": (3, 0, 0),
    "category": "Object",
    "author": "Mitch",
    "version": (0, 3),
    "description": "Tools to randomize and render MTG-like text for OCR training"
}

import bpy
import random
import math
import string
import os
import re
import bpy_extras
import bmesh
import mathutils
from mathutils import Vector
import copy

ALLOWED_CHARS = string.ascii_letters + string.digits + "' ,"

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def generate_random_text(max_char):
    length = random.randint(1, max_char)
    return ''.join(random.choice(ALLOWED_CHARS) for _ in range(length))

def apply_random_text(obj, max_char):
    obj.data.body = generate_random_text(max_char)
    return obj.data.body

def apply_text(obj):
    settings = bpy.context.scene.mtg_settings  # type: ignore
    obj.data.body = settings.text_input
    return obj.data.body

def apply_random_rotation(obj, max_angle):
    obj.rotation_euler = (
        math.radians(random.uniform(-max_angle, max_angle)),
        math.radians(random.uniform(-max_angle, max_angle)),
        math.radians(random.uniform(-max_angle, max_angle))
    )
    return obj.rotation_euler

def apply_random_size(obj, max_size):
    random_size = random.uniform(0.2, max_size)
    obj.scale = (random_size, random_size, random_size)
    return obj.scale

def make_text_mesh_copy(text_obj, dbg_col, base):
    # Duplicate the object
    bpy.ops.object.select_all(action='DESELECT')
    text_obj.select_set(True)
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object

    # Convert duplicate to mesh
    bpy.ops.object.convert(target='MESH')
    return dup

def sanitize_basename(name: str) -> str:
    # Keep letters, digits, space, apostrophe, underscore, dash. Strip leading/trailing spaces.
    name = name.strip()
    return re.sub(r"[^A-Za-z0-9 _'\-]", "_", name)

def ensure_unique_basename(out_dir: str, base: str) -> str:
    # Avoid overwriting when random text repeats
    candidate = base
    i = 1
    while os.path.exists(os.path.join(out_dir, candidate + ".png")) or \
          os.path.exists(os.path.join(out_dir, candidate + ".gt.txt")):
        candidate = f"{base}_{i:03d}"
        i += 1
    return candidate

def write_gt_text(out_dir: str, base: str, text: str) -> None:
    # Tesseract trains on *.gt.txt files (line-level GT)
    path = os.path.join(out_dir, base + ".gt.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

def extract_glyphs(mesh_copy, dbg_col, min_verts=3, min_size=0.001):
    """
    Extract individual glyphs from a mesh copy of a text object.
    Filters out empty or tiny fragments.

    Args:
        mesh_copy: Blender mesh object (copy of the text converted to mesh)
        dbg_col: Blender collection to store debug glyph objects
        min_verts: Minimum vertices to consider a valid glyph
        min_size: Minimum world-space size (largest axis) to consider valid
    Returns:
        List of glyph objects
    """
    bpy.context.view_layer.objects.active = mesh_copy
    mesh_copy.select_set(True)
    glyphs = []

    while len(mesh_copy.data.vertices) > 0:
        # Deselect all in EDIT mode
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        verts = mesh_copy.data.vertices
        if not verts:
            break

        # Pick right-most vertex (X) to start glyph selection
        right_vert = max(verts, key=lambda v: v.co.x)
        right_vert.select = True
        existing_objs = set(bpy.context.scene.objects)
        # Select all linked vertices for this glyph
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_linked(delimit={'SEAM'})
        bpy.ops.mesh.separate(type='SELECTED')
        bpy.ops.object.mode_set(mode='OBJECT')

        # Identify the newly separated object
        new_objs = [o for o in bpy.context.scene.objects if o not in existing_objs]
        if new_objs:
            obj = new_objs[0]

        # Compute actual bounding box size in world space
        coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
        if coords:
            xs = [c.x for c in coords]
            ys = [c.y for c in coords]
            zs = [c.z for c in coords]
            bbox_size = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
        else:
            bbox_size = 0.0

        # Filter out tiny / empty fragments
        if len(obj.data.vertices) >= min_verts and bbox_size >= min_size:
            glyphs.append(obj)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)

        # Reset active object to the remaining mesh
        bpy.context.view_layer.objects.active = mesh_copy

    return glyphs

def sort_glyphs_left_to_right(glyphs):
    def glyph_center_x(obj):
        coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
        return sum(c.x for c in coords) / len(coords)
    return sorted(glyphs, key=glyph_center_x)

def merge_close_meshes(mesh_objs, thresh=0.05): #not used anymore
    """
    Merge meshes in `mesh_objs` whose center-x are within `thresh`.
    Returns a new list of objects (some joined).
    """
    merged = []
    used = set()

    for i, obj_a in enumerate(mesh_objs):
        if obj_a in used:
            continue

        coords_a = [obj_a.matrix_world @ v.co for v in obj_a.data.vertices]
        center_a = sum(v.x for v in coords_a) / len(coords_a)

        group = [obj_a]

        for j, obj_b in enumerate(mesh_objs):
            if obj_b in used or obj_b == obj_a:
                continue
            coords_b = [obj_b.matrix_world @ v.co for v in obj_b.data.vertices]
            center_b = sum(v.x for v in coords_b) / len(coords_b)

            if abs(center_a - center_b) < thresh:
                group.append(obj_b)
                used.add(obj_b)

        if len(group) > 1:
            # Make sure objects are selected for joining
            bpy.ops.object.select_all(action='DESELECT')
            for g in group:
                g.select_set(True)
            bpy.context.view_layer.objects.active = group[0]
            bpy.ops.object.join()  # merges into group[0]
            merged.append(group[0])
        else:
            merged.append(obj_a)

        used.add(obj_a)

    return merged
def merge_close_glyphs(glyphs, text_scale):
    """
    Merge glyphs that are horizontally close.

    Args:
        glyphs (list[bpy.types.Object]): original glyph objects
        threshold (float): merging distance threshold
        text_scale: Vector or similar, used for scaling threshold

    Returns:
        list[bpy.types.Object]: merged glyph objects (originals intact until merged)
    """
    #remaining = glyphs.copy()  # shallow copy to iterate safely
    merged_list = []
    remaining = [g.name for g in glyphs if g and g.name in bpy.data.objects]
    print("Remaining contents:", remaining)
    print("Remaining types:", [type(x) for x in remaining])

    while remaining:
        obj_name = remaining.pop(0)
        print("obj", obj_name)
        print("obj type", type(obj_name))

        obj_a = bpy.data.objects[obj_name]
        coords_a = [obj_a.matrix_world @ v.co for v in obj_a.data.vertices]
        min_x_a = min(v.x for v in coords_a)
        max_x_a = max(v.x for v in coords_a)
        center_x_a = (min_x_a + max_x_a) / 2

        merge_group = [obj_a]

        # Compare to every other glyph in remaining
        for obj_name_2 in remaining[:remaining.index(obj_name)]:
            obj_b = bpy.data.objects[obj_name_2]
            print("obj", obj_name_2)
            print("obj type", type(obj_name_2))
            coords_b = [obj_b.matrix_world @ v.co for v in obj_b.data.vertices]
            min_x_b = min(v.x for v in coords_b)
            max_x_b = max(v.x for v in coords_b)
            center_x_b = (min_x_b + max_x_b) / 2

            if abs(center_x_a - center_x_b) < bpy.context.scene.mtg_settings.threshold * text_scale.y: # type: ignore
                merge_group.append(obj_b)

        # If more than one glyph in merge_group, join into obj_a
        if len(merge_group) > 1:
            bpy.context.view_layer.objects.active = obj_a
            for o in merge_group:
                o.select_set(True)
            bpy.ops.object.join()  # obj_a becomes the merged glyph

        merged_list.append(obj_a)

    return merged_list
def merge_close_glyphs_V2(glyphs, text_scale):
    merged_list = []

    # STEP 1: Build groups (list of lists)
    groups = {}
    for g in glyphs:
        if g and g.name in bpy.data.objects:
            obj = bpy.data.objects[g.name]
            center_x = sum((v.co.x for v in obj.data.vertices)) / len(obj.data.vertices)
            group_id = round(center_x / (bpy.context.scene.mtg_settings.threshold * text_scale.y))  # type: ignore
            groups.setdefault(group_id, []).append(g.name)

    print("Groups:", groups)

    # STEP 2: Merge each group separately
    for gid, names in groups.items():
        group_objs = [bpy.data.objects[n] for n in names if n in bpy.data.objects]
        if not group_objs:
            continue

        if len(group_objs) > 1:
            main_obj = group_objs[0]
            bpy.context.view_layer.objects.active = main_obj
            for o in group_objs:
                if o.name in bpy.data.objects:
                    o.select_set(True)
            bpy.ops.object.join()
            merged_list.append(main_obj)
        else:
            merged_list.append(group_objs[0])

    return merged_list


def write_glyph_box_V3(file_handle, glyph_obj, cam, scene, char, res_x, res_y, dbg_col, index, text_scale, prev_glyphs=None):
    if prev_glyphs is None:
        prev_glyphs = []

    # Compute world coordinates
    world_coords = [glyph_obj.matrix_world @ v.co for v in glyph_obj.data.vertices]

    # Compute horizontal center
    min_x = min(v.x for v in world_coords)
    max_x = max(v.x for v in world_coords)
    center_x = (min_x + max_x) / 2

    # Check all previous glyphs for merge
    merge_candidates = []
    for prev_obj, prev_center_x in prev_glyphs:
        if abs(prev_center_x - center_x) < bpy.context.scene.mtg_settings.threshold * text_scale.y: # type: ignore
            merge_candidates.append((prev_obj, prev_center_x))

    if merge_candidates:
        for prev_obj, prev_center_x in merge_candidates:
            world_coords += [prev_obj.matrix_world @ v.co for v in prev_obj.data.vertices]
            bpy.data.objects.remove(prev_obj, do_unlink=True)
            prev_glyphs.remove((prev_obj, prev_center_x))

    # Project to camera
    projs = [bpy_extras.object_utils.world_to_camera_view(scene, cam, wc) for wc in world_coords]
    us = [p.x for p in projs]
    vs = [p.y for p in projs]

    u_min, u_max = max(0.0, min(us)), min(1.0, max(us))
    v_min, v_max = max(0.0, min(vs)), min(1.0, max(vs))

    x1, x2 = sorted([int(round(u_min * res_x)), int(round(u_max * res_x))])
    y1, y2 = sorted([int(round(v_min * res_y)), int(round(v_max * res_y))])

    # Brute force: extend vertical box for 'i' and 'j'
    if char in {"i", "j"}:
        pad = int(round(0.5 * text_scale.y * res_y))  # tweak factor as needed
        #y1 = max(0, y1 - pad)
        y2 = min(res_y, y2 + pad)

    # Write box line
    file_handle.write(f"{char} {x1} {y1} {x2} {y2} 0\n")

    # Debug wireframe box
    min_x_w, max_x_w = min(v.x for v in world_coords), max(v.x for v in world_coords)
    min_y_w, max_y_w = min(v.y for v in world_coords), max(v.y for v in world_coords)
    z_w = min(v.z for v in world_coords)

    box_verts = [(min_x_w, min_y_w, z_w),
                 (max_x_w, min_y_w, z_w),
                 (max_x_w, max_y_w, z_w),
                 (min_x_w, max_y_w, z_w)]
    box_edges = [(0,1),(1,2),(2,3),(3,0)]
    box_mesh = bpy.data.meshes.new(f"DBG_box_{index}_{char}")
    box_mesh.from_pydata(box_verts, box_edges, [])
    box_obj = bpy.data.objects.new(f"DBG_box_{index}_{char}", box_mesh)
    dbg_col.objects.link(box_obj)
    box_obj.display_type = 'WIRE'
    box_obj.show_in_front = True

    # Track this glyph for potential merges with next ones
    prev_glyphs.append((box_obj, center_x))
def write_glyph_box_V4(path, merged_sorted_glyphs, chars,  cam, scene, res_x, res_y, dbg_col, text_scale):
    with open(path, "w", encoding="utf-8") as f:
        for i, glyph_obj in enumerate(merged_sorted_glyphs):
            if i < len(chars):
                # Compute world coordinates
                world_coords = [glyph_obj.matrix_world @ v.co for v in glyph_obj.data.vertices]

                # Project to camera
                projs = [bpy_extras.object_utils.world_to_camera_view(scene, cam, wc) for wc in world_coords]
                us = [p.x for p in projs]
                vs = [p.y for p in projs]

                u_min, u_max = max(0.0, min(us)), min(1.0, max(us))
                v_min, v_max = max(0.0, min(vs)), min(1.0, max(vs))

                x1, x2 = sorted([int(round(u_min * res_x)), int(round(u_max * res_x))])
                y1, y2 = sorted([int(round(v_min * res_y)), int(round(v_max * res_y))])

                # Brute force: extend vertical box for 'i' and 'j'
                if chars[i] in {"i", "j"}:
                    pad = int(round(0.5 * text_scale.y * res_y))  # tweak factor as needed
                    #y1 = max(0, y1 - pad)
                    y2 = min(res_y, y2 + pad)

                # Write box line
                file_handle.write(f"{chars[i]} {x1} {y1} {x2} {y2} 0\n") # type: ignore

                # Debug wireframe box
                min_x_w, max_x_w = min(v.x for v in world_coords), max(v.x for v in world_coords)
                min_y_w, max_y_w = min(v.y for v in world_coords), max(v.y for v in world_coords)
                z_w = min(v.z for v in world_coords)

                box_verts = [(min_x_w, min_y_w, z_w),
                             (max_x_w, min_y_w, z_w),
                             (max_x_w, max_y_w, z_w),
                             (min_x_w, max_y_w, z_w)]
                box_edges = [(0,1),(1,2),(2,3),(3,0)]
                box_mesh = bpy.data.meshes.new(f"DBG_box_{i}_{chars[i]}")
                box_mesh.from_pydata(box_verts, box_edges, [])
                box_obj = bpy.data.objects.new(f"DBG_box_{i}_{chars[i]}", box_mesh)
                dbg_col.objects.link(box_obj)
                box_obj.display_type = 'WIRE'
                box_obj.show_in_front = True

def write_box_file_debug_V3(out_dir: str, base: str, mesh_obj, cam, scene, dbg_col):
    """
    Debug version (mesh-only):
      - Takes a mesh copy
      - Splits into loose parts (glyphs)
      - Writes boxfile
      - Spawns debug geometry
    """

    final_res_x = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    final_res_y = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

    path = os.path.join(out_dir, base + ".box")
    chars = list(base)  # you already pass `base` from text input

    # Split into loose parts
    bpy.context.view_layer.objects.active = mesh_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    parts = [o for o in dbg_col.objects if o.name.startswith(mesh_obj.name)]
    parts = sorted(parts, key=lambda o: min(v.co.x for v in o.data.vertices))

    with open(path, "w", encoding="utf-8") as f:
        for i, part in enumerate(parts):
            if i >= len(chars):
                break
            c = chars[i]

            # World coords
            world_coords = [part.matrix_world @ v.co for v in part.data.vertices]

            # Camera projection
            projs = [bpy_extras.object_utils.world_to_camera_view(scene, cam, wc) for wc in world_coords]
            us = [p.x for p in projs]
            vs = [p.y for p in projs]

            u_min = max(0.0, min(us))
            u_max = min(1.0, max(us))
            v_min = max(0.0, min(vs))
            v_max = min(1.0, max(vs))

            x1 = int(round(u_min * final_res_x))
            x2 = int(round(u_max * final_res_x))
            y1 = int(round(v_min * final_res_y))
            y2 = int(round(v_max * final_res_y))

            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            f.write(f"{c} {x1} {y1} {x2} {y2} 0\n")

            # Debug wire box
            min_x_w = min(v.x for v in world_coords)
            max_x_w = max(v.x for v in world_coords)
            min_y_w = min(v.y for v in world_coords)
            max_y_w = max(v.y for v in world_coords)
            z_w = min(v.z for v in world_coords)

            box_verts = [
                (min_x_w, min_y_w, z_w),
                (max_x_w, min_y_w, z_w),
                (max_x_w, max_y_w, z_w),
                (min_x_w, max_y_w, z_w),
            ]
            box_edges = [(0,1),(1,2),(2,3),(3,0)]
            box_mesh = bpy.data.meshes.new(f"DBG_box_{i}_{c}")
            box_mesh.from_pydata(box_verts, box_edges, [])
            box_mesh.update()

            box_obj = bpy.data.objects.new(f"DBG_box_{i}_{c}", box_mesh)
            dbg_col.objects.link(box_obj)
            box_obj.display_type = 'WIRE'
            box_obj.show_in_front = True
def write_box_file_debug_V4(out_dir: str, base: str, mesh_copy, cam, scene, dbg_col, text_scale):

    res_x = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    res_y = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

    path = os.path.join(out_dir, base + ".box")
    chars = list(mesh_copy.data.body if hasattr(mesh_copy.data, "body") else base)
    glyphs = []
    glyphs = extract_glyphs(mesh_copy, dbg_col)
    sorted_glyphs = []
    sorted_glyphs = sort_glyphs_left_to_right(glyphs)
    merged_sorted_glyphs = []
    merged_sorted_glyphs = merge_close_glyphs_V2(sorted_glyphs, text_scale)
    #write_glyph_box_V4(path, merged_sorted_glyphs, chars, cam, scene, res_x, res_y, dbg_col, text_scale)

# ----------------------------------------------------------
# Property group
# ----------------------------------------------------------
class MTG_OT_Settings(bpy.types.PropertyGroup):
    max_rotation: bpy.props.FloatProperty(
        name="Max Rotation (°)",
        description="Maximum random rotation applied on each axis",
        default=15.0,
        min=0.0,
        max=90.0
    ) # type: ignore
    max_size: bpy.props.FloatProperty(
        name="Max Size",
        description="Maximum size applied to text obj",
        default=1,
        min=0.01,
        max=2
    ) # type: ignore
    max_characters: bpy.props.IntProperty(
        name="Max Characters",
        description="Maximum number of characters",
        default=10,
        min=1,
        max=15
    ) # type: ignore
    text_input: bpy.props.StringProperty(
        name="Text",
        description="Enter the text to use",
        default="Hello iijjIi"
    ) # type: ignore
    num_renders: bpy.props.IntProperty(
        name="Number of Renders",
        description="How many renders to generate",
        default=5,
        min=1,
        max=1000
    ) # type: ignore
    output_dir: bpy.props.StringProperty(
        name="Output Directory",
        description="Where to save renders",
        subtype='DIR_PATH',
        default="//ocr_renders/"
    ) # type: ignore
    threshold: bpy.props.FloatProperty(
        name="Threshold",
        description="Threshold for merging",
        default=0.5,
        min=0.005,
        max=3
    ) # type: ignore
# ----------------------------------------------------------
# Operators
# ----------------------------------------------------------
class MTG_OT_RandomizeRotation(bpy.types.Operator):
    bl_idname = "mtg.randomize_rotation"
    bl_label = "Randomize Rotation"

    def execute(self, context): # type: ignore
        settings = context.scene.mtg_settings  # type: ignore
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}

        rot = apply_random_rotation(obj, settings.max_rotation)
        self.report({'INFO'}, f"Rotation applied: {rot}")
        return {'FINISHED'}
class MTG_OT_Randomize_text(bpy.types.Operator):
    bl_idname = "mtg.randomize_text"
    bl_label = "Randomize Text"

    def execute(self, context): # type: ignore
        settings = context.scene.mtg_settings # type: ignore
        obj = context.active_object
        if not obj or obj.type != "FONT":
            self.report({'WARNING'}, "Select a Text object first")
            return {'CANCELLED'}

        new_text = apply_random_text(obj, settings.max_characters)
        self.report({'INFO'}, f"Text set to: {new_text}")
        return {'FINISHED'}
class MTG_OT_GiveString(bpy.types.Operator):
    bl_idname = "mtg.give_string"
    bl_label = "Give String"
    bl_description = "Input a custom string for box file testing"

    def execute(self, context): # type: ignore
        # Get active object (make sure it's a text object)
        settings = context.scene.mtg_settings # type: ignore
        obj = context.active_object
        if obj and obj.type == 'FONT':
            obj.data.body = settings.text_input
            self.report({'INFO'}, f"Updated text to: {settings.text_input}")
        else:
            self.report({'WARNING'}, "Select a Text object first")
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)
class MTG_OT_Randomize_size(bpy.types.Operator):
    bl_idname = "mtg.randomize_size"
    bl_label = "Randomize Size"

    def execute(self, context): # type: ignore
        settings = context.scene.mtg_settings # type: ignore
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}

        scale = apply_random_size(obj, settings.max_size)
        self.report({'INFO'}, f"Scale applied: {scale}")
        return {'FINISHED'}
class MTG_OT_BatchRender(bpy.types.Operator):
    bl_idname = "mtg.batch_render"
    bl_label = "Batch Render"

    def execute(self, context): # type: ignore
        settings = context.scene.mtg_settings # type: ignore
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        cam = context.scene.camera

        for i in range(settings.num_renders):
            new_text = apply_random_text(obj, settings.max_characters)
            apply_random_rotation(obj, settings.max_rotation)
            apply_random_size(obj, settings.max_size)

            base = sanitize_basename(new_text)
            base = ensure_unique_basename(out_dir, base)

            context.scene.render.filepath = os.path.join(out_dir, base + ".png")
            bpy.ops.render.render(write_still=True)

            write_gt_text(out_dir, base, new_text)
            #write_box_file(out_dir, base, obj, cam, context.scene, True) #last param given is the debug, make sure it's set to False when generating moe than 1 render
            write_box_file_debug_V3(out_dir, base, obj, cam, context.scene)  # full debugging for write_box_file

            self.report({'INFO'}, f"Rendered {base}.png, wrote {base}.gt.txt and {base}.box")

        return {'FINISHED'}
class MTG_OT_Render(bpy.types.Operator):
    bl_idname = "mtg.render"
    bl_label = "Render"

    def execute(self, context): # type: ignore
        settings = context.scene.mtg_settings # type: ignore
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        cam = context.scene.camera

        new_text = apply_text(obj)
        apply_random_rotation(obj, settings.max_rotation)
        text_scale = apply_random_size(obj, settings.max_size)

        base = sanitize_basename(new_text)
        base = ensure_unique_basename(out_dir, base)

        context.scene.render.filepath = os.path.join(out_dir, base + ".png")
        bpy.ops.render.render(write_still=True)

        write_gt_text(out_dir, base, new_text)
        #write_box_file(out_dir, base, obj, cam, context.scene, True) #last param given is the debug, make sure it's set to False when generating moe than 1 render
        #write_box_file_debug(out_dir, base, obj, cam, context.scene)  # full debugging for write_box_file #note2: Not using this anymore.
        dbg_col_name = f"DBG_{base}"
        if dbg_col_name in bpy.data.collections:
            dbg_col = bpy.data.collections[dbg_col_name]
        else:
            dbg_col = bpy.data.collections.new(dbg_col_name)
            context.scene.collection.children.link(dbg_col)

        mesh_copy = make_text_mesh_copy(obj, dbg_col, base)
        write_box_file_debug_V4(out_dir, base, mesh_copy, cam, context.scene, dbg_col, text_scale)

        self.report({'INFO'}, f"Rendered {base}.png, wrote {base}.gt.txt and {base}.box")

        return {'FINISHED'}

# ----------------------------------------------------------
# Operator: Draw Box File (Preview)
# ----------------------------------------------------------
class MTG_OT_DrawBoxFile(bpy.types.Operator):
    bl_idname = "mtg.draw_box_file"
    bl_label = "Draw Box File"
    bl_description = "Draws bounding boxes from a Tesseract .box file in the scene"

    filepath: bpy.props.StringProperty( 
        name="Box File",
        description="Path to the .box file",
        subtype='FILE_PATH'
    ) # type: ignore
    box_height: bpy.props.FloatProperty(
        name="Z Height",
        description="Height to place boxes in Z axis",
        default=0.0
    ) # type: ignore
    line_thickness: bpy.props.FloatProperty(
        name="Line Thickness",
        default=0.01
    ) # type: ignore

    def execute(self, context):
        if not self.filepath or not os.path.exists(self.filepath):
            self.report({'ERROR'}, "Invalid file path")
            return {'CANCELLED'}

        scene = context.scene
        cam = scene.camera
        if not cam:
            self.report({'ERROR'}, "No camera in scene")
            return {'CANCELLED'}

        res_x = scene.render.resolution_x
        res_y = scene.render.resolution_y

        # Get orthographic scale / dimensions
        if cam.data.type != 'ORTHO':
            self.report({'WARNING'}, "Camera is not orthographic; mapping may be off")
        ortho_scale = cam.data.ortho_scale

        # Camera world location
        cam_loc = cam.location

        # Create collection
        col_name = "BoxFilePreview"
        if col_name in bpy.data.collections:
            col = bpy.data.collections[col_name]
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            col = bpy.data.collections.new(col_name)
            scene.collection.children.link(col)

        with open(self.filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                char, x1, y1, x2, y2 = parts[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Map pixel coords to camera XY
                # Blender orthographic top-down: center at camera location, scale = ortho_scale
                def pixel_to_world(px, py):
                    wx = (px / res_x - 0.5) * ortho_scale + cam_loc.x
                    wy = (py / res_y - 0.5) * ortho_scale + cam_loc.y
                    return wx, wy

                # wx1, wy1 = pixel_to_world(x1, res_y - y1)  # flip Y
                # wx2, wy2 = pixel_to_world(x2, res_y - y2)  # flip Y
                wx1, wy1 = pixel_to_world(x1, y1)  # not flip Y
                wx2, wy2 = pixel_to_world(x2, y2)  # not flip Y

                verts = [
                    (wx1, wy1, self.box_height),
                    (wx2, wy1, self.box_height),
                    (wx2, wy2, self.box_height),
                    (wx1, wy2, self.box_height)
                ]
                edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
                mesh = bpy.data.meshes.new(f"Box_{char}")
                mesh.from_pydata(verts, edges, [])
                mesh.update()

                obj = bpy.data.objects.new(f"Box_{char}", mesh)
                col.objects.link(obj)
                obj.display_type = 'WIRE'
                obj.show_in_front = True

        self.report({'INFO'}, f"Boxes drawn from {self.filepath}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}
# ----------------------------------------------------------
# Panel
# ----------------------------------------------------------
class MTG_PT_MainPanel(bpy.types.Panel):
    bl_label = "OCR Text Generator"
    bl_idname = "MTG_PT_mainpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OCR Trainer'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtg_settings  # type: ignore


        layout.prop(settings, "max_rotation")
        layout.prop(settings, "max_size")
        layout.prop(settings, "max_characters")
        layout.prop(settings, "text_input")
        layout.prop(settings, "num_renders")
        layout.prop(settings, "output_dir")
        layout.prop(settings, "filepath")
        layout.prop(settings, "box_height")
        layout.prop(settings, "line_thickness")
        layout.prop(settings, "threshold")

        layout.operator("mtg.randomize_rotation")
        layout.operator("mtg.randomize_text")
        layout.operator("mtg.randomize_size")
        layout.operator("mtg.draw_box_file")
        layout.operator("mtg.render")
        layout.operator("mtg.batch_render")
# ----------------------------------------------------------
# Register
# ----------------------------------------------------------
classes = [
    MTG_OT_Settings,
    MTG_OT_RandomizeRotation,
    MTG_OT_Randomize_text,
    MTG_OT_GiveString,
    MTG_OT_Randomize_size,
    MTG_OT_Render,
    MTG_OT_BatchRender,
    MTG_OT_DrawBoxFile,
    MTG_PT_MainPanel
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mtg_settings = bpy.props.PointerProperty(type=MTG_OT_Settings) # type: ignore

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mtg_settings # type: ignore

if __name__ == "__main__":
    register()
