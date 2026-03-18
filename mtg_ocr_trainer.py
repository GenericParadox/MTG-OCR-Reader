bl_info = {
    "name": "MTG OCR Trainer Tools",
    "blender": (3, 0, 0),
    "category": "Object",
    "author": "Mitch",
    "version": (1, 0),
    "description": "Randomize and render MTG-like text for OCR training with boxfile generation",
}

import bpy
import random
import math
import string
import os
import re
import bpy_extras
from mathutils import Vector

# ---------------------------
# Config
# ---------------------------
ALLOWED_CHARS = string.ascii_letters + string.digits + "' ,"
DEFAULT_OUTPUT = "//ocr_renders/"


# ---------------------------
# Utilities
# ---------------------------

def sanitize_basename(name: str) -> str:
    name = name.strip()
    return re.sub(r"[^A-Za-z0-9 _'\-]", "_", name) or "text"


def ensure_unique_basename(out_dir: str, base: str) -> str:
    candidate = base
    i = 1
    while (os.path.exists(os.path.join(out_dir, candidate + ".png")) or
           os.path.exists(os.path.join(out_dir, candidate + ".gt.txt")) or
           os.path.exists(os.path.join(out_dir, candidate + ".box"))):
        candidate = f"{base}_{i:03d}"
        i += 1
    return candidate


def write_gt_text(out_dir: str, base: str, text: str) -> None:
    path = os.path.join(out_dir, base + ".gt.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text + "\n")


def generate_random_text(max_char: int) -> str:
    length = random.randint(1, max_char)
    return ''.join(random.choice(ALLOWED_CHARS) for _ in range(length))


def apply_random_rotation(obj, max_angle: float):
    obj.rotation_euler = (
        math.radians(random.uniform(-max_angle, max_angle)),
        math.radians(random.uniform(-max_angle, max_angle)),
        math.radians(random.uniform(-max_angle, max_angle)),
    )


def apply_random_size(obj, max_size: float) -> float:
    scale = random.uniform(0.2, max_size)
    obj.scale = (scale, scale, scale)
    return scale


# ---------------------------
# Mesh conversion & glyph extraction
# ---------------------------

def make_text_mesh_copy(text_obj: bpy.types.Object, base_name: str) -> bpy.types.Object:
    """Duplicate a font object, convert the duplicate to mesh, and return it."""
    bpy.ops.object.select_all(action='DESELECT')
    text_obj.select_set(True)
    bpy.context.view_layer.objects.active = text_obj
    bpy.ops.object.duplicate()
    dup = bpy.context.active_object
    dup.name = f"{text_obj.name}_MESH_{base_name}"
    bpy.ops.object.convert(target='MESH')
    return bpy.context.active_object


def get_world_coords(obj: bpy.types.Object):
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


def get_bbox(obj: bpy.types.Object):
    coords = get_world_coords(obj)
    xs = [c.x for c in coords]
    ys = [c.y for c in coords]
    return min(xs), max(xs), min(ys), max(ys)


def get_center_x(obj: bpy.types.Object) -> float:
    x_min, x_max, _, _ = get_bbox(obj)
    return (x_min + x_max) / 2


def get_height(obj: bpy.types.Object) -> float:
    _, _, y_min, y_max = get_bbox(obj)
    return abs(y_max - y_min)


def extract_glyphs(
    mesh_obj: bpy.types.Object,
    dbg_col: bpy.types.Collection,
    debug_mode: bool,
    min_verts: int = 3,
    min_size: float = 1e-4,
) -> list:
    """
    Separate a mesh into loose parts (individual glyphs).
    Tiny/degenerate parts are discarded.
    Returns list of glyph objects.
    """
    existing = set(bpy.data.objects.keys())

    bpy.context.view_layer.objects.active = mesh_obj
    mesh_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.separate(type='LOOSE')
    bpy.ops.object.mode_set(mode='OBJECT')

    new_objs = [mesh_obj] + [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in existing]
    glyphs = []

    for obj in new_objs:
        if obj.type != 'MESH':
            continue

        coords = get_world_coords(obj)
        if not coords:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue

        xs = [c.x for c in coords]
        ys = [c.y for c in coords]
        zs = [c.z for c in coords]
        bbox_dim = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

        if len(obj.data.vertices) < min_verts or bbox_dim < min_size:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue

        if debug_mode and dbg_col and obj.name not in dbg_col.objects:
            dbg_col.objects.link(obj)

        glyphs.append(obj)

    return glyphs


# ---------------------------
# Glyph merging
# ---------------------------

def merge_close_glyphs(glyphs: list, text_scale: float, threshold: float, dot_ratio: float = 0.40) -> list:
    """
    Merge dot glyphs (i/j dots) into their parent stem glyph.

    Two glyphs are merged only when BOTH conditions are met:
      1. Their X centers are within (threshold * text_scale) of each other.
      2. The smaller glyph's height is less than dot_ratio of the larger one,
         i.e. one is clearly a dot, not a full letter.

    This prevents adjacent skinny letters (l, i, i) from being incorrectly merged,
    since they are similar in height even when close in X.
    """
    if not glyphs:
        return []

    scale = text_scale if isinstance(text_scale, (int, float)) else 1.0
    x_threshold = threshold * scale

    sorted_glyphs = sorted(glyphs, key=get_center_x)
    groups = []
    used = set()

    for i, g in enumerate(sorted_glyphs):
        if i in used:
            continue

        group = [g]
        cx_i = get_center_x(g)
        h_i = get_height(g)

        for j, other in enumerate(sorted_glyphs):
            if j <= i or j in used:
                continue

            cx_j = get_center_x(other)
            h_j = get_height(other)

            if abs(cx_i - cx_j) > x_threshold:
                continue

            h_big = max(h_i, h_j)
            h_small = min(h_i, h_j)

            if h_big < 1e-6:
                continue

            if (h_small / h_big) < dot_ratio:
                group.append(other)
                used.add(j)

        used.add(i)
        groups.append(group)

    merged = []
    for group in groups:
        if len(group) == 1:
            merged.append(group[0])
            continue

        bpy.ops.object.select_all(action='DESELECT')
        for obj in group:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = group[0]
        try:
            bpy.ops.object.join()
            merged.append(bpy.context.view_layer.objects.active)
        except RuntimeError:
            merged.extend(group)

    merged.sort(key=get_center_x)
    print("merged glyphs:", merged)
    return merged


# ---------------------------
# Box file writing
# ---------------------------

def write_box_file(
    path: str,
    merged_glyphs: list,
    chars: list,
    cam,
    scene,
    res_x: int,
    res_y: int,
    dbg_col,
    
):
    """
    Write a Tesseract .box file.
    Format per line: <char> x1 y1 x2 y2 0
    Coordinates are projected from world space via the camera.
    """
    if not cam:
        raise RuntimeError("No camera provided to box writer")

    def project(world_coords):
        projs = [bpy_extras.object_utils.world_to_camera_view(scene, cam, wc) for wc in world_coords]
        us = [p.x for p in projs]
        vs = [p.y for p in projs]
        x1 = int(round(max(0.0, min(us)) * res_x))
        x2 = int(round(min(1.0, max(us)) * res_x))
        #y1 = int(round(max(0.0, min(vs)) * res_y)) axis seems flipped
        #y2 = int(round(min(1.0, max(vs)) * res_y))
        y1 = int(round((1.0 - min(1.0, max(vs))) * res_y))
        y2 = int(round((1.0 - max(0.0, min(vs))) * res_y))
        x1, x2 = sorted([max(0, min(res_x - 1, x1)), max(0, min(res_x - 1, x2))])
        y1, y2 = sorted([max(0, min(res_y - 1, y1)), max(0, min(res_y - 1, y2))])
        return x1, y1, x2, y2

    with open(path, "w", encoding="utf-8") as f:
        for i, glyph_obj in enumerate(merged_glyphs):
            if i >= len(chars):
                print(f"Warning: more glyphs ({len(merged_glyphs)}) than chars ({len(chars)}), stopping at {i}")
                break

            ch = chars[i]
            world_coords = get_world_coords(glyph_obj)
            x1, y1, x2, y2 = project(world_coords)
            f.write(f"{ch} {x1} {y1} {x2} {y2} 0\n")

            if dbg_col:
                _draw_debug_box(world_coords, i, ch, dbg_col)


def _draw_debug_box(world_coords, idx, ch, dbg_col):
    """Draw a wireframe bounding box in world space directly from glyph verts."""
    xs = [c.x for c in world_coords]
    ys = [c.y for c in world_coords]
    zs = [c.z for c in world_coords]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    z_val = sum(zs) / len(zs)  # average Z, sits on the glyph plane

    verts = [
        (x_min, y_min, z_val),
        (x_max, y_min, z_val),
        (x_max, y_max, z_val),
        (x_min, y_max, z_val),
    ]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]

    mesh = bpy.data.meshes.new(f"DBG_box_{idx}_{ch}")
    mesh.from_pydata(verts, edges, [])
    mesh.update()
    obj = bpy.data.objects.new(f"DBG_box_{idx}_{ch}", mesh)
    dbg_col.objects.link(obj)
    obj.display_type = 'WIRE'
    obj.show_in_front = True

# ---------------------------
# Shared render helpers
# ---------------------------

def get_or_create_debug_collection(scene, name: str) -> bpy.types.Collection:
    if name in bpy.data.collections:
        col = bpy.data.collections[name]
        for obj in list(col.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
    else:
        col = bpy.data.collections.new(name)
        scene.collection.children.link(col)
    return col


def do_render(context, text_obj, text: str, settings, out_dir: str):
    """
    Core render routine shared by single and batch operators.
    Applies text/transform, renders, writes .gt.txt and .box.
    Returns the base filename used.
    """
    scene = context.scene
    cam = scene.camera

    obj = text_obj
    obj.data.body = text
    apply_random_rotation(obj, settings.max_rotation)
    text_scale = apply_random_size(obj, settings.max_size)

    base = sanitize_basename(text)
    base = ensure_unique_basename(out_dir, base)

    scene.render.filepath = os.path.join(out_dir, base + ".png")
    bpy.ops.render.render(write_still=True)
    write_gt_text(out_dir, base, text)

    dbg_col = None
    if settings.debug_mode:
        dbg_col = get_or_create_debug_collection(scene, f"DBG_{base}")

    res_x = int(scene.render.resolution_x * scene.render.resolution_percentage / 100)
    res_y = int(scene.render.resolution_y * scene.render.resolution_percentage / 100)

    mesh_copy = make_text_mesh_copy(obj, base)
    glyphs = extract_glyphs(mesh_copy, dbg_col, settings.debug_mode)
    merged = merge_close_glyphs(glyphs, text_scale, settings.threshold, settings.dot_ratio)
    chars = list(text)

    box_path = os.path.join(out_dir, base + ".box")
    write_box_file(box_path, merged, chars, cam, scene, res_x, res_y, dbg_col)

     # Cleanup glyph meshes
    for g in merged:
        bpy.data.objects.remove(g, do_unlink=True)

    return base


# ---------------------------
# Property Group
# ---------------------------

class MTG_Settings(bpy.types.PropertyGroup):
    max_rotation: bpy.props.FloatProperty(
        name="Max Rotation (°)", default=15.0, min=0.0, max=90.0
    )  # type: ignore
    max_size: bpy.props.FloatProperty(
        name="Max Size", default=1.0, min=0.01, max=3.0
    )  # type: ignore
    max_characters: bpy.props.IntProperty(
        name="Max Characters", default=10, min=1, max=32
    )  # type: ignore
    text_input: bpy.props.StringProperty(
        name="Text", default="Hello iijjIi"
    )  # type: ignore
    num_renders: bpy.props.IntProperty(
        name="Number of Renders", default=5, min=1, max=10000
    )  # type: ignore
    output_dir: bpy.props.StringProperty(
        name="Output Directory", subtype='DIR_PATH', default=DEFAULT_OUTPUT
    )  # type: ignore
    threshold: bpy.props.FloatProperty(
        name="X Merge Threshold", default=0.5, min=0.005, max=5.0
    )  # type: ignore
    dot_ratio: bpy.props.FloatProperty(
        name="Dot Ratio", default=0.40, min=0.05, max=0.95,
        description="Max height ratio for a glyph to be considered a dot (i/j). Lower = stricter."
    )  # type: ignore
    debug_mode: bpy.props.BoolProperty(
        name="Debug Mode (create geometry)", default=True
    )  # type: ignore


# ---------------------------
# Operators
# ---------------------------

class MTG_OT_RandomizeRotation(bpy.types.Operator):
    bl_idname = "mtg.randomize_rotation"
    bl_label = "Randomize Rotation"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        apply_random_rotation(obj, context.scene.mtg_settings.max_rotation)
        return {'FINISHED'}


class MTG_OT_RandomizeText(bpy.types.Operator):
    bl_idname = "mtg.randomize_text"
    bl_label = "Randomize Text"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        new_text = generate_random_text(context.scene.mtg_settings.max_characters)
        obj.data.body = new_text
        self.report({'INFO'}, f"Text set to: {new_text}")
        return {'FINISHED'}


class MTG_OT_RandomizeSize(bpy.types.Operator):
    bl_idname = "mtg.randomize_size"
    bl_label = "Randomize Size"

    def execute(self, context):
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        apply_random_size(obj, context.scene.mtg_settings.max_size)
        return {'FINISHED'}


class MTG_OT_Render(bpy.types.Operator):
    bl_idname = "mtg.render"
    bl_label = "Render (single)"

    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        cam = context.scene.camera

        if not obj or obj.type != 'FONT':
            self.report({'ERROR'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        if not cam:
            self.report({'ERROR'}, "No active camera in scene")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        base = do_render(context, obj, settings.text_input, settings, out_dir)
        self.report({'INFO'}, f"Rendered {base}.png + .gt.txt + .box")
        return {'FINISHED'}


class MTG_OT_BatchRender(bpy.types.Operator):
    bl_idname = "mtg.batch_render"
    bl_label = "Batch Render"

    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        cam = context.scene.camera

        if not obj or obj.type != 'FONT':
            self.report({'ERROR'}, "Select a text object (FONT) first")
            return {'CANCELLED'}
        if not cam:
            self.report({'ERROR'}, "No active camera in scene")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(settings.num_renders):
            text = generate_random_text(settings.max_characters)
            do_render(context, obj, text, settings, out_dir)
            print(f"Batch progress: {i + 1}/{settings.num_renders}")

        self.report({'INFO'}, f"Batch rendered {settings.num_renders} images to {out_dir}")
        return {'FINISHED'}


class MTG_OT_DrawBoxFile(bpy.types.Operator):
    bl_idname = "mtg.draw_box_file"
    bl_label = "Draw Box File"
    filepath: bpy.props.StringProperty(name="Box File", subtype='FILE_PATH')  # type: ignore

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

        with open(self.filepath, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                ch, x1, y1, x2, y2 = parts[0], *map(int, parts[1:5])
                _draw_debug_box(None, idx, ch, x1, y1, x2, y2, res_x, res_y, cam, col)

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

        layout.label(text="Transform Settings")
        layout.prop(settings, "max_rotation")
        layout.prop(settings, "max_size")

        layout.separator()
        layout.label(text="Text Settings")
        layout.prop(settings, "max_characters")
        layout.prop(settings, "text_input")

        layout.separator()
        layout.label(text="Output Settings")
        layout.prop(settings, "num_renders")
        layout.prop(settings, "output_dir")

        layout.separator()
        layout.label(text="Merge Settings")
        layout.prop(settings, "threshold")
        layout.prop(settings, "dot_ratio")

        layout.separator()
        layout.prop(settings, "debug_mode")

        layout.separator()
        layout.label(text="Manual Controls")
        layout.operator("mtg.randomize_rotation")
        layout.operator("mtg.randomize_text")
        layout.operator("mtg.randomize_size")

        layout.separator()
        layout.label(text="Render")
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
    bpy.types.Scene.mtg_settings = bpy.props.PointerProperty(type=MTG_Settings)  # type: ignore


def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    try:
        del bpy.types.Scene.mtg_settings  # type: ignore
    except Exception:
        pass


if __name__ == "__main__":
    register()
