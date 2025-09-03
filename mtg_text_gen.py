bl_info = {
    "name": "MTG OCR Trainer Tools",
    "blender": (3, 0, 0),
    "category": "Object",
    "author": "Mitch",
    "version": (0, 2),
    "description": "Tools to randomize and render MTG-like text for OCR training"
}

import bpy
import random
import math
import string
import os

ALLOWED_CHARS = string.ascii_letters + string.digits + "' "

def generate_random_text(max_char):
    length = random.randint(1, max_char)
    return ''.join(random.choice(ALLOWED_CHARS) for _ in range(length))

# Property group for settings
class MTG_OT_Settings(bpy.types.PropertyGroup):
    max_rotation: bpy.props.FloatProperty(
        name="Max Rotation (°)",
        description="Maximum random rotation applied on each axis",
        default=15.0,
        min=0.0,
        max=90.0
    )
    max_size: bpy.props.FloatProperty(
        name = "Max Size",
        description = "Maximum size applied to text obj",
        default = 1,
        min = 0.01,
        max = 2
    )
    max_characters: bpy.props.FloatProperty(
        name = "Max Character",
        description = "Maximum nr of characters",
        default = 10,
        min = 1,
        max = 15
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

# Operator: randomize rotation
class MTG_OT_RandomizeRotation(bpy.types.Operator):
    bl_idname = "mtg.randomize_rotation"
    bl_label = "Randomize Rotation"
    bl_description = "Randomly rotate the selected text object"

    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        max_angle = settings.max_rotation
        if obj is None or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}


        obj.rotation_euler = (
            math.radians(random.uniform(-max_angle, max_angle)),
            math.radians(random.uniform(-max_angle, max_angle)),
            math.radians(random.uniform(-max_angle, max_angle))
        )

        self.report({'INFO'}, f"Rotation applied: {obj.rotation_euler}")
        return {'FINISHED'}

# Operator: randomize text
class OBJECT_OT_randomize_text(bpy.types.Operator):
    bl_idname = "object.randomize_text"
    bl_label = "Randomize Text"

    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object
        max_char = settings.max_characters
        if obj and obj.type == "FONT":
            obj.data.body = generate_random_text(max_char)
            self.report({'INFO'}, f"Text set to: {obj.data.body}")
        else:
            self.report({'WARNING'}, "Select a Text object first")
        return {'FINISHED'}


# Operator: randomize size
class OBJECT_OT_randomize_size(bpy.types.Operator):
    bl_idname = "object.randomize_size"
    bl_label = "Randomize Size"

    def execute(self, context):
        obj = context.active_object
        settings = context.scene.mtg_settings
        max_size = settings.max_size
        random_size = random.uniform(0.2, max_size) #0.2 is minimum allowable scale as to not run into negatives
        if obj is None or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}

        obj.scale = (
            random_size,
            random_size,
            random_size
        )
        self.report({'INFO'}, f"Rotation applied: {obj.rotation_euler}")
        return {'FINISHED'}

# Operator: batch render
class MTG_OT_BatchRender(bpy.types.Operator):
    bl_idname = "mtg.batch_render"
    bl_label = "Batch Render"
    bl_description = "Generate random text, apply random rotation, and render multiple images"

    def execute(self, context):
        settings = context.scene.mtg_settings
        obj = context.active_object

        if obj is None or obj.type != 'FONT':
            self.report({'WARNING'}, "Select a text object first!")
            return {'CANCELLED'}

        out_dir = bpy.path.abspath(settings.output_dir)
        os.makedirs(out_dir, exist_ok=True)

        for i in range(settings.num_renders):
            # Random text + rotation
            max_char = settings.max_characters
            obj.data.body = generate_random_text(max_char)
            max_angle = settings.max_rotation
            obj.rotation_euler = (
                math.radians(random.uniform(-max_angle, max_angle)),
                math.radians(random.uniform(-max_angle, max_angle)),
                math.radians(random.uniform(-max_angle, max_angle))
            )
            max_size = settings.max_size
            random_size = random.uniform(0.2, max_size) #0.2 is minimum allowable scale as to not run into negatives
            obj.scale = (
                random_size,
                random_size,
                random_size
            )

            # Set output filepath
            filename = f"{obj.data.body}.png"
            bpy.context.scene.render.filepath = os.path.join(out_dir, filename)

            # Render and save
            bpy.ops.render.render(write_still=True)

            self.report({'INFO'}, f"Rendered {filename}")

        return {'FINISHED'}

# UI Panel
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

        layout.operator("mtg.randomize_rotation")
        layout.operator("object.randomize_text", text="Randomize Text")
        layout.operator("object.randomize_size", text = "Randomize Size")
        layout.operator("mtg.batch_render", text="Batch Render")


# Register/unregister
classes = [MTG_OT_Settings, MTG_OT_RandomizeRotation, OBJECT_OT_randomize_text, MTG_OT_BatchRender, OBJECT_OT_randomize_size, MTG_PT_MainPanel]

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
