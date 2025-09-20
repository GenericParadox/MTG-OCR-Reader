bl_info = {
    "name": "MTG OCR Trainer Tools (Fixed Single-File)",
    "blender": (3, 0, 0),
    "category": "Object",
    "author": "Mitch (cleaned by assistant)",
    "version": (0, 9),
    "description": "Randomize and render MTG-like text for OCR training with boxfile generation",
}

import bpy
import random
import math
import string
import os
import re
import bpy_extras
import mathutils
from mathutils import Vector

# ---------------------------
# Config / constants
# ---------------------------
ALLOWED_CHARS = string.ascii_letters + string.digits + "' ,"
DEFAULT_OUTPUT = "//ocr_renders/"

# ---------------------------
# Utilities
# ---------------------------
def sanitize_basename(name: str) -> str:
    name = name.strip()
    # keep letters, numbers, space, apostrophe, underscore, dash
    return re.sub(r"[^A-Za-z0-9 _'\-]", "_", name) or "text"

def ensure_unique_basename(out_dir: str, base: str) -> str:
    candidate = base
    i = 1
    while os.path.exists(os.path.join(out_dir, candidate + ".png")) or \
          os.path.exists(os.path.join(out_dir, candidate + ".gt.txt")) or \
          os.path.exists(os.path.join(out_dir, candidate + ".box")):
        candidate = f"{base}_{i:03d}"
        i += 1
    return candidate

def write_gt_text(out_dir: str, base: str, text: str) -> None:
    path = os.path.join(out_dir, base + ".gt.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

def generate_random_text(max_char):
    length = random.randint(1, max_char)
    return ''.join(random.choice(ALLOWED_CHARS) for _ in range(length))

def apply_random_rotation(obj, max_angle):
    obj.rotation_euler = (
        math.radians(random.uniform(-max_angle, max_angle)),
        math.radians(random.uniform(-max_angle, max_angle)),
        math.radians(random.uniform(-max_angle, max_angle))
    )
    return obj.rotation_euler

def apply_random_size(obj, max_size):
    random_size = random.uniform(0.2, max_size)
    vec = Vector((random_size, random_size, random_size))
    obj.scale = vec
    return vec

# ---------------------------
# Mesh/text conversion & glyph extraction
# ---------------------------
def make_text_mesh_copy(text_obj: bpy.types.Object, dbg_col: bpy.types.Collection, base_name: str):
    # duplicate text object and convert duplicate to mesh, place into dbg_col
    bpy.ops.object.select_all(action='DESELECT')
    text_obj.select_set(True)
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup_name = f"{text_obj.name}_MESH_{base_name}"
    dup.name = dup_name

    # Convert to mesh
    bpy.ops.object.convert(target='MESH')
    mesh_obj = bpy.context.active_object
    mesh_obj.name = dup_name

    # Ensure it's linked to the debug collection for later easy filtering
    if dbg_col and mesh_obj.name not in dbg_col.objects:
        dbg_col.objects.link(mesh_obj)

    return mesh_obj

def make_text_mesh_copy(text_obj: bpy.types.Object, dbg_col: bpy.types.Collection, base_name: str):
    # duplicate text object and convert duplicate to mesh
    bpy.ops.object.select_all(action='DESELECT')
    text_obj.select_set(True)
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object

    dup_name = f"{text_obj.name}_MESH_{base_name}"
    dup.name = dup_name

    # Convert to mesh
    bpy.ops.object.convert(target='MESH')
    mesh_obj = bpy.context.active_object
    mesh_obj.name = dup_name

    # Unlink from all other collections to prevent render/selection conflicts
    for coll in list(mesh_obj.users_collection):
        coll.objects.unlink(mesh_obj)

    # Link only to debug collection if provided
    if dbg_col:
        dbg_col.objects.link(mesh_obj)

    return mesh_obj


def extract_glyphs(mesh_obj: bpy.types.Object, dbg_col: bpy.types.Collection, min_verts=3, min_size=1e-4):
    """
    - Use mesh_obj (a mesh) -> separate loose parts -> collect new objects created
    - Returns list of objects representing glyph pieces (world transforms preserved)
    """
    scene = bpy.context.scene
    existing = set(bpy.data.objects.keys())

    # Ensure the mesh_obj is active and in object mode
    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')  # operate on whole mesh
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    new_objs = [mesh_obj] + [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in existing]
    print("new_objs:     ", new_objs)
    glyphs = []
    for o in new_objs:
        # Only keep mesh objects that are not empty
        print("Extracted Glyph: ", o, " type: ", o.type)
        if o.type != 'MESH':
            continue
        # Link to dbg_col
        if dbg_col and o.name not in dbg_col.objects:
            dbg_col.objects.link(o)
        # compute bbox size in world coords
        coords = [o.matrix_world @ v.co for v in o.data.vertices] if o.data.vertices else []
        if not coords:
            bpy.data.objects.remove(o, do_unlink=True)
            continue
        xs = [c.x for c in coords]; ys = [c.y for c in coords]; zs = [c.z for c in coords]
        bbox_dim = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
        if len(o.data.vertices) >= min_verts and bbox_dim >= min_size:
            glyphs.append(o)
        else:
            bpy.data.objects.remove(o, do_unlink=True)
            

    return glyphs

def glyph_center_x(obj: bpy.types.Object):
    coords = [obj.matrix_world @ v.co for v in obj.data.vertices]
    return sum(c.x for c in coords) / len(coords)

def sort_glyphs_left_to_right(glyphs):
    print("sorted:", sorted(glyphs, key=glyph_center_x))
    return sorted(glyphs, key=glyph_center_x)

def merge_close_glyphs_V2(glyphs, text_scale_world, threshold):
    """
    Group glyphs by world-space X buckets based on threshold * text_scale_world (float).
    Returns merged objects; joins groups when needed.
    """
    if not glyphs:
        return []

    # Build groups by bucket
    groups = {}
    bucket_width = max(1e-6, threshold * (text_scale_world if isinstance(text_scale_world, (int, float)) else getattr(text_scale_world, "y", 1.0)))
    for g in glyphs:
        if g.name not in bpy.data.objects:
            continue
        ws = [g.matrix_world @ v.co for v in g.data.vertices]
        cx = sum(v.x for v in ws) / len(ws)
        gid = int(round(cx / bucket_width))
        groups.setdefault(gid, []).append(g)

    merged = []
    for gid, objs in groups.items():
        if len(objs) == 1:
            merged.append(objs[0])
            continue
        # Join them
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        try:
            bpy.ops.object.join()
            merged.append(bpy.context.view_layer.objects.active)
        except RuntimeError:
            # Joining might fail if objects aren't in same collection/linked properly - fallback keep originals
            merged.extend(objs)

    # Sort left to right
    merged.sort(key=glyph_center_x)
    print("merged: ", merged)
    return merged

# ---------------------------
# Box writing
# ---------------------------
def write_box_file_v4(path, merged_sorted_glyphs, chars, cam, scene, res_x, res_y, dbg_col, text_scale_world):
    """
    Writes tesseract-style .box with lines: <char> x1 y1 x2 y2 0
    Coordinates use Blender's world_to_camera_view normalized coords -> multiplied by res.
    """
    if not cam:
        raise RuntimeError("No camera provided to box writer")

    def project_world_coords(world_coords):
        projs = [bpy_extras.object_utils.world_to_camera_view(scene, cam, wc) for wc in world_coords]
        us = [p.x for p in projs]
        vs = [p.y for p in projs]
        u_min, u_max = max(0.0, min(us)), min(1.0, max(us))
        v_min, v_max = max(0.0, min(vs)), min(1.0, max(vs))
        x1 = int(round(u_min * res_x))
        x2 = int(round(u_max * res_x))
        y1 = int(round(v_min * res_y))
        y2 = int(round(v_max * res_y))
        # clamp
        x1 = max(0, min(res_x - 1, x1))
        x2 = max(0, min(res_x - 1, x2))
        y1 = max(0, min(res_y - 1, y1))
        y2 = max(0, min(res_y - 1, y2))
        # ensure ordering
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        return x1, y1, x2, y2

    with open(path, "w", encoding="utf-8") as f:
        for i, glyph_obj in enumerate(merged_sorted_glyphs):
            print("GlyphPair: ", i," with ", glyph_obj)
            #if i > len(chars)+1:
            #    break
            ch = chars[i]
            world_coords = [glyph_obj.matrix_world @ v.co for v in glyph_obj.data.vertices]
            x1, y1, x2, y2 = project_world_coords(world_coords)

            # extend vertical box for i/j to reduce cropped dots
            #if ch in {"i", "j"}:
            #    pad = int(round(0.02 * res_y + 0.5 * (getattr(text_scale_world, "y", text_scale_world) * res_y)))
            #    y2 = min(res_y - 1, y2 + pad)

            f.write(f"{ch} {x1} {y1} {x2} {y2} 0\n")

            # optional debug box as wireframe object
            min_x_w = min(v.x for v in world_coords)
            max_x_w = max(v.x for v in world_coords)
            min_y_w = min(v.y for v in world_coords)
            max_y_w = max(v.y for v in world_coords)
            z_w = min(v.z for v in world_coords)
            box_verts = [(min_x_w, min_y_w, z_w),
                         (max_x_w, min_y_w, z_w),
                         (max_x_w, max_y_w, z_w),
                         (min_x_w, max_y_w, z_w)]
            box_edges = [(0,1),(1,2),(2,3),(3,0)]
            box_mesh = bpy.data.meshes.new(f"DBG_box_old_{i}_{ch}")
            box_mesh.from_pydata(box_verts, box_edges, [])
            box_mesh.update()
            box_obj = bpy.data.objects.new(f"DBG_box_old_{i}_{ch}", box_mesh)
            if dbg_col:
                dbg_col.objects.link(box_obj)
            box_obj.display_type = 'WIRE'
            box_obj.show_in_front = True
            if dbg_col:
                # back to normalized coords
                u1, u2 = x1 / res_x, x2 / res_x
                v1, v2 = y1 / res_y, y2 / res_y

                # put them in camera space at some Z depth
                depth = 0.1  # just in front of glyphs
                verts_cam = [
                    (u1*2-1, v1*2-1, -depth),  # bottom left (NDC -> [-1,1])
                    (u2*2-1, v1*2-1, -depth),
                    (u2*2-1, v2*2-1, -depth),
                    (u1*2-1, v2*2-1, -depth),
                ]

                # transform camera-space verts into world space
                verts_world = [cam.matrix_world @ Vector(v) for v in verts_cam]

                # build debug mesh
                box_edges = [(0,1),(1,2),(2,3),(3,0)]
                box_mesh = bpy.data.meshes.new(f"DBG_box_{i}_{ch}")
                box_mesh.from_pydata(verts_world, box_edges, [])
                box_obj = bpy.data.objects.new(f"DBG_box_{i}_{ch}", box_mesh)
                dbg_col.objects.link(box_obj)
                box_obj.display_type = 'WIRE'
                box_obj.show_in_front = True

# ---------------------------
# Property Group
# ---------------------------
class MTG_Settings(bpy.types.PropertyGroup):
    max_rotation: bpy.props.FloatProperty(
        name="Max Rotation (°)",
        default=15.0,
        min=0.0,
        max=90.0
    ) # type: ignore
    max_size: bpy.props.FloatProperty(
        name="Max Size",
        default=1.0,
        min=0.01,
        max=3.0
    ) # type: ignore
    max_characters: bpy.props.IntProperty(
        name="Max Characters",
        default=10,
        min=1,
        max=32
    ) # type: ignore
    text_input: bpy.props.StringProperty(
        name="Text",
        default="Hello iijjIi"
    ) # type: ignore
    num_renders: bpy.props.IntProperty(
        name="Number of Renders",
        default=5,
        min=1,
        max=10000
    ) # type: ignore
    output_dir: bpy.props.StringProperty(
        name="Output Directory",
        subtype='DIR_PATH',
        default=DEFAULT_OUTPUT
    ) # type: ignore
    threshold: bpy.props.FloatProperty(
        name="Threshold",
        default=0.5,
        min=0.005,
        max=5.0
    ) # type: ignore
    debug_mode: bpy.props.BoolProperty(
        name="Debug Mode (create geometry)",
        default=True
    ) # type: ignore

# ---------------------------
# Operators
# ---------------------------
class MTG_OT_RandomizeRotation(bpy.types.Operator):
    bl_idname = "mtg.randomize_rotation"
    bl_label = "Randomize Rotation"
    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        rot = apply_random_rotation(obj, settings.max_rotation)
        self.report({'INFO'}, f"Rotation applied: {rot}")
        return {'FINISHED'}

class MTG_OT_RandomizeText(bpy.types.Operator):
    bl_idname = "mtg.randomize_text"
    bl_label = "Randomize Text"
    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        new_text = generate_random_text(settings.max_characters)
        obj.data.body = new_text
        self.report({'INFO'}, f"Text set to: {new_text}")
        return {'FINISHED'}

class MTG_OT_RandomizeSize(bpy.types.Operator):
    bl_idname = "mtg.randomize_size"
    bl_label = "Randomize Size"
    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        scale = apply_random_size(obj, settings.max_size)
        self.report({'INFO'}, f"Scale applied: {scale}")
        return {'FINISHED'}

class MTG_OT_Render(bpy.types.Operator):
    bl_idname = "mtg.render"
    bl_label = "Render (single)"
    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        scene = context.scene
        cam = scene.camera
        if not obj or obj.type != 'FONT':
            self.report({'ERROR'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        if not cam:
            self.report({'ERROR'}, "No active camera in scene")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        # Apply provided text, random transform & size
        new_text = settings.text_input
        obj.data.body = new_text
        apply_random_rotation(obj, settings.max_rotation)
        text_scale = apply_random_size(obj, settings.max_size)

        base = sanitize_basename(new_text)
        base = ensure_unique_basename(out_dir, base)
        scene.render.filepath = os.path.join(out_dir, base + ".png")
        bpy.ops.render.render(write_still=True)

        write_gt_text(out_dir, base, new_text)

        # Setup debug collection if needed
        dbg_col = None
        if settings.debug_mode:
            dbg_col_name = f"DBG_{base}"
            if dbg_col_name in bpy.data.collections:
                dbg_col = bpy.data.collections[dbg_col_name]
                # clear old
                for o in list(dbg_col.objects):
                    bpy.data.objects.remove(o, do_unlink=True)
            else:
                dbg_col = bpy.data.collections.new(dbg_col_name)
                scene.collection.children.link(dbg_col)

        # Make mesh copy and produce box
        mesh_copy = make_text_mesh_copy_V2(obj, dbg_col, base)
        res_x = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
        res_y = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
        glyphs = extract_glyphs(mesh_copy, dbg_col)
        sorted_glyphs = sort_glyphs_left_to_right(glyphs)
        merged = merge_close_glyphs_V2(sorted_glyphs, text_scale, settings.threshold)
        chars = list(new_text)
        box_path = os.path.join(out_dir, base + ".box")
        write_box_file_v4(box_path, merged, chars, cam, scene, res_x, res_y, dbg_col if settings.debug_mode else None, text_scale)

        self.report({'INFO'}, f"Rendered {base}.png and wrote {base}.gt.txt and {base}.box")
        return {'FINISHED'}

class MTG_OT_BatchRender(bpy.types.Operator):
    bl_idname = "mtg.batch_render"
    bl_label = "Batch Render"
    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        scene = context.scene
        cam = scene.camera
        if not obj or obj.type != 'FONT':
            self.report({'ERROR'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        if not cam:
            self.report({'ERROR'}, "No active camera in scene")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(settings.num_renders):
            new_text = generate_random_text(settings.max_characters)
            obj.data.body = new_text
            apply_random_rotation(obj, settings.max_rotation)
            text_scale = apply_random_size(obj, settings.max_size)

            base = sanitize_basename(new_text)
            base = ensure_unique_basename(out_dir, base)
            scene.render.filepath = os.path.join(out_dir, base + ".png")
            bpy.ops.render.render(write_still=True)
            write_gt_text(out_dir, base, new_text)

            dbg_col = None
            if settings.debug_mode:
                dbg_col_name = f"DBG_{base}"
                if dbg_col_name in bpy.data.collections:
                    dbg_col = bpy.data.collections[dbg_col_name]
                    for o in list(dbg_col.objects):
                        bpy.data.objects.remove(o, do_unlink=True)
                else:
                    dbg_col = bpy.data.collections.new(dbg_col_name)
                    scene.collection.children.link(dbg_col)

            mesh_copy = make_text_mesh_copy(obj, dbg_col, base)
            res_x = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
            res_y = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)
            glyphs = extract_glyphs(mesh_copy, dbg_col)
            sorted_glyphs = sort_glyphs_left_to_right(glyphs)
            merged = merge_close_glyphs_V2(sorted_glyphs, text_scale, settings.threshold)
            chars = list(new_text)
            box_path = os.path.join(out_dir, base + ".box")
            write_box_file_v4(box_path, merged, chars, cam, scene, res_x, res_y, dbg_col if settings.debug_mode else None, text_scale)

        self.report({'INFO'}, f"Batch rendered {settings.num_renders} images to {out_dir}")
        return {'FINISHED'}

class MTG_OT_DrawBoxFile(bpy.types.Operator):
    bl_idname = "mtg.draw_box_file"
    bl_label = "Draw Box File"
    filepath: bpy.props.StringProperty(name="Box File", subtype='FILE_PATH') # type: ignore
    box_height: bpy.props.FloatProperty(name="Z Height", default=0.0) # type: ignore
    line_thickness: bpy.props.FloatProperty(name="Line Thickness", default=0.01) # type: ignore

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

        col_name = "BoxFilePreview"
        if col_name in bpy.data.collections:
            col = bpy.data.collections[col_name]
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            col = bpy.data.collections.new(col_name)
            scene.collection.children.link(col)

        # simple mapping: normalized coords -> ortho/ortho-scale mapping is hard; we'll place boxes in camera view plane approx.
        with open(self.filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                char, x1, y1, x2, y2 = parts[:5]
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Map pixel coords to normalized 0..1
                u1, v1 = x1 / res_x, y1 / res_y
                u2, v2 = x2 / res_x, y2 / res_y

                # Build simple plane in world space at box_height (this is approximate visual aid)
                z = self.box_height
                verts = [(u1, v1, z), (u2, v1, z), (u2, v2, z), (u1, v2, z)]
                edges = [(0,1),(1,2),(2,3),(3,0)]
                mesh = bpy.data.meshes.new(f"BoxPreview_{idx}_{char}")
                mesh.from_pydata(verts, edges, [])
                obj = bpy.data.objects.new(f"BoxPreview_{idx}_{char}", mesh)
                col.objects.link(obj)
                obj.display_type = 'WIRE'
                obj.show_in_front = True

        self.report({'INFO'}, f"Boxes drawn from {self.filepath}")
        return {'FINISHED'}

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

# ---------------------------
# UI Panel
# ---------------------------
class MTG_PT_MainPanel(bpy.types.Panel):
    bl_label = "OCR Text Generator"
    bl_idname = "MTG_PT_mainpanel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'OCR Trainer'

    def draw(self, context):
        layout = self.layout
        settings = context.scene.mtg_settings

        layout.prop(settings, "max_rotation")
        layout.prop(settings, "max_size")
        layout.prop(settings, "max_characters")
        layout.prop(settings, "text_input")
        layout.prop(settings, "num_renders")
        layout.prop(settings, "output_dir")
        layout.prop(settings, "threshold")
        layout.prop(settings, "debug_mode")

        layout.separator()
        layout.operator("mtg.randomize_rotation")
        layout.operator("mtg.randomize_text")
        layout.operator("mtg.randomize_size")
        layout.separator()
        layout.operator("mtg.render")
        layout.operator("mtg.batch_render")
        layout.separator()
        layout.operator("mtg.draw_box_file")

# ---------------------------
# Registration
# ---------------------------
classes = (
    MTG_Settings,
    MTG_OT_RandomizeRotation,
    MTG_OT_RandomizeText,
    MTG_OT_RandomizeSize,
    MTG_OT_Render,
    MTG_OT_BatchRender,
    MTG_OT_DrawBoxFile,
    MTG_PT_MainPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mtg_settings = bpy.props.PointerProperty(type=MTG_Settings) # type: ignore

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    try:
        del bpy.types.Scene.mtg_settings # type: ignore
    except Exception:
        pass

if __name__ == "__main__":
    register()

# ---------------------------
# Quick test snippets (copy-paste into Blender Text Editor and run)
# ---------------------------
"""
# 1) Create a sample font object and camera
import bpy
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.text_add(location=(0,0,0))
txt = bpy.context.active_object
txt.data.body = "Test i j"
# set font size and extrude to 0 so it's flat
txt.data.extrude = 0
txt.scale = (1.0, 1.0, 1.0)

# Create camera
bpy.ops.object.camera_add(location=(0, -5, 0), rotation=(math.radians(90), 0, 0))
cam = bpy.context.active_object
bpy.context.scene.camera = cam

# Ensure render settings sane
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.context.scene.render.resolution_percentage = 100

# Configure addon settings
bpy.context.scene.mtg_settings.output_dir = "//ocr_test/"
bpy.context.scene.mtg_settings.debug_mode = True

# 2) Run a single render operator from Python
bpy.ops.mtg.render()

# 3) Or run batch
bpy.context.scene.mtg_settings.num_renders = 5
bpy.ops.mtg.batch_render()
"""
