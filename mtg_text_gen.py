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
from mathutils import Vector

ALLOWED_CHARS = string.ascii_letters + string.digits + "' "

# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------
def generate_random_text(max_char):
    length = random.randint(1, max_char)
    return ''.join(random.choice(ALLOWED_CHARS) for _ in range(length))

def apply_random_text(obj, max_char):
    obj.data.body = generate_random_text(max_char)
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

def write_box_file(out_dir: str, base: str, text: str, original_obj, cam, scene):
    path = os.path.join(out_dir, base + ".box")
    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    # Start horizontal cursor at original object location
    cursor_x = original_obj.location.x
    cursor_y = original_obj.location.y  # assuming top-down orthographic, Y stays constant

    with open(path, "w", encoding="utf-8") as f:
        for ch in text:
            if ch == " ":
                # Move cursor by space width dynamically
                tmp = original_obj.copy()
                tmp.data = original_obj.data.copy()
                tmp.data.body = "A"  # measure a typical letter
                scene.collection.objects.link(tmp)

                depsgraph = bpy.context.evaluated_depsgraph_get()
                eval_obj = tmp.evaluated_get(depsgraph)
                mesh_data = eval_obj.to_mesh()
                coords = [tmp.matrix_world @ v.co for v in mesh_data.vertices]
                space_width = max(v.x for v in coords) - min(v.x for v in coords)
                cursor_x += space_width  # advance cursor
                tmp.to_mesh_clear()
                bpy.data.objects.remove(tmp, do_unlink=True)
                continue

            # Duplicate character
            tmp = original_obj.copy()
            tmp.data = original_obj.data.copy()
            tmp.data.body = ch
            tmp.location.x = cursor_x
            tmp.location.y = cursor_y
            scene.collection.objects.link(tmp)

            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = tmp.evaluated_get(depsgraph)
            mesh_data = eval_obj.to_mesh()

            coords = [tmp.matrix_world @ v.co for v in mesh_data.vertices]
            min_x = min(v.x for v in coords)
            max_x = max(v.x for v in coords)
            min_y = min(v.y for v in coords)
            max_y = max(v.y for v in coords)

            # Project to camera
            bl = bpy_extras.object_utils.world_to_camera_view(scene, cam, Vector((min_x, min_y, 0)))
            tr = bpy_extras.object_utils.world_to_camera_view(scene, cam, Vector((max_x, max_y, 0)))

            # Convert normalized to pixel coordinates
            x1 = int(bl.x * res_x)
            #y1 = res_y - int(bl.y * res_y)
            y1 = int(bl.y * rex_y)
            x2 = int(tr.x * res_x)
            #y2 = res_y - int(tr.y * res_y)
            y2 = int(tr.y * res_y)

            # Ensure proper order
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])

            f.write(f"{ch} {x1} {y1} {x2} {y2} 0\n")

            # Advance cursor for next character
            char_width = max_x - min_x
            cursor_x += char_width

            # Cleanup
            tmp.to_mesh_clear()
            bpy.data.objects.remove(tmp, do_unlink=True)



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
    )
    max_size: bpy.props.FloatProperty(
        name="Max Size",
        description="Maximum size applied to text obj",
        default=1,
        min=0.01,
        max=2
    )
    max_characters: bpy.props.IntProperty(
        name="Max Characters",
        description="Maximum number of characters",
        default=10,
        min=1,
        max=15
    )
    num_renders: bpy.props.IntProperty(
        name="Number of Renders",
        description="How many renders to generate",
        default=5,
        min=1,
        max=1000
    )
    output_dir: bpy.props.StringProperty(
        name="Output Directory",
        description="Where to save renders",
        subtype='DIR_PATH',
        default="//ocr_renders/"
    )

# ----------------------------------------------------------
# Operators
# ----------------------------------------------------------
class MTG_OT_RandomizeRotation(bpy.types.Operator):
    bl_idname = "mtg.randomize_rotation"
    bl_label = "Randomize Rotation"

    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        if not obj or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}

        rot = apply_random_rotation(obj, settings.max_rotation)
        self.report({'INFO'}, f"Rotation applied: {rot}")
        return {'FINISHED'}

class OBJECT_OT_randomize_text(bpy.types.Operator):
    bl_idname = "object.randomize_text"
    bl_label = "Randomize Text"

    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        if not obj or obj.type != "FONT":
            self.report({'WARNING'}, "Select a Text object first")
            return {'CANCELLED'}

        new_text = apply_random_text(obj, settings.max_characters)
        self.report({'INFO'}, f"Text set to: {new_text}")
        return {'FINISHED'}

class OBJECT_OT_randomize_size(bpy.types.Operator):
    bl_idname = "object.randomize_size"
    bl_label = "Randomize Size"

    def execute(self, context):
        settings = context.scene.mtg_settings
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

    def execute(self, context):
        settings = context.scene.mtg_settings
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
            write_box_file(out_dir, base, new_text, obj, cam, context.scene)

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
    )
    box_height: bpy.props.FloatProperty(
        name="Z Height",
        description="Height to place boxes in Z axis",
        default=0.0
    )
    line_thickness: bpy.props.FloatProperty(
        name="Line Thickness",
        default=0.01
    )

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
        settings = context.scene.mtg_settings

        layout.prop(settings, "max_rotation")
        layout.prop(settings, "max_size")
        layout.prop(settings, "max_characters")
        layout.prop(settings, "num_renders")
        layout.prop(settings, "output_dir")
        layout.prop(settings, "filepath")
        layout.prop(settings, "box_height")
        layout.prop(settings, "line_thickness")

        layout.operator("mtg.randomize_rotation")
        layout.operator("object.randomize_text")
        layout.operator("object.randomize_size")
        layout.operator("mtg.draw_box_file")
        layout.operator("mtg.batch_render")

# ----------------------------------------------------------
# Register
# ----------------------------------------------------------
classes = [
    MTG_OT_Settings,
    MTG_OT_RandomizeRotation,
    OBJECT_OT_randomize_text,
    OBJECT_OT_randomize_size,
    MTG_OT_BatchRender,
    MTG_OT_DrawBoxFile,
    MTG_PT_MainPanel
]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mtg_settings = bpy.props.PointerProperty(type=MTG_OT_Settings)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    del bpy.types.Scene.mtg_settings

if __name__ == "__main__":
    register()
