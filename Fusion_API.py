# Author: ChatGPT + Jonas :)
# Description:
# Create 300 randomly placed cylinders (4 mm diameter, 20 mm length)
# in a 7 x 7 x 7 cm cube using the Fusion 360 API.

import adsk.core, adsk.fusion, adsk.cam, traceback
import random

def run(context):
    ui = None
    try:
        app = adsk.core.Application.get()
        ui  = app.userInterface

        product = app.activeProduct
        design = adsk.fusion.Design.cast(product)
        if not design:
            ui.messageBox('No active Fusion design. Please open a design first.')
            return

        rootComp = design.rootComponent

        # ---------------------------------------------------------------------
        # PARAMETERS
        # ---------------------------------------------------------------------
        num_rods   = 300

        side_cm    = 7.0          # cube edge length in cm
        length_cm  = 2.0          # cylinder length in cm
        dia_mm     = 4.0          # diameter in mm

        # Fusion database units are centimeters.
        radius_cm  = (dia_mm / 10.0) / 2.0   # 4 mm -> 0.4 cm -> radius = 0.2 cm

        half_side  = side_cm / 2.0

        # Optional: fix random seed for reproducibility
        # random.seed(0)

        # ---------------------------------------------------------------------
        # CREATE BASE CYLINDER COMPONENT (template)
        # ---------------------------------------------------------------------
        occs = rootComp.occurrences
        base_transform = adsk.core.Matrix3D.create()
        rod_occ = occs.addNewComponent(base_transform)
        rod_comp = rod_occ.component
        rod_comp.name = 'Rod_4mm_x_20mm'

        # Sketch a circle on the XY plane at the origin
        sketches = rod_comp.sketches
        xy_plane = rod_comp.xYConstructionPlane
        sketch = sketches.add(xy_plane)

        circles = sketch.sketchCurves.sketchCircles
        center_point = adsk.core.Point3D.create(0, 0, 0)
        circles.addByCenterRadius(center_point, radius_cm)

        # Use the circle profile to create an extrusion (the cylinder)
        prof = sketch.profiles.item(0)
        extrudes = rod_comp.features.extrudeFeatures

        ext_input = extrudes.createInput(
            prof,
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation
        )

        # Define an extrusion of "length_cm" in +Z direction.
        distance = adsk.core.ValueInput.createByReal(length_cm)
        # False -> one-sided, True -> symmetric (here one-sided is fine)
        ext_input.setDistanceExtent(False, distance)
        ext_input.isSolid = True
        extrudes.add(ext_input)

        # ---------------------------------------------------------------------
        # PLACE 300 OCCURRENCES RANDOMLY IN THE CUBE
        # ---------------------------------------------------------------------
        #
        # Cube is centered at (0,0,0), so coordinates go from -half_side to +half_side.
        #
        # Cylinder is defined from z = 0 to z = length_cm.
        # We will shift the *bottom* of the cylinder to z_base:
        #   z_base is chosen such that: -half_side <= z_base <= half_side - length_cm
        #
        # For x and y, we keep a margin of radius_cm so rods stay inside the cube walls.

        for i in range(num_rods):
            x = random.uniform(-half_side + radius_cm, half_side - radius_cm)
            y = random.uniform(-half_side + radius_cm, half_side - radius_cm)
            z_base = random.uniform(-half_side, half_side - length_cm)

            # Build transform for this rod occurrence
            trans = adsk.core.Matrix3D.create()
            # Set translation directly in the transform matrix.
            trans.setCell(0, 3, x)
            trans.setCell(1, 3, y)
            trans.setCell(2, 3, z_base)

            if i == 0:
                # Reuse the original occurrence as the first rod
                rod_occ.transform = trans
            else:
                # Create additional occurrences of the same component
                occs.addExistingComponent(rod_comp, trans)

        ui.messageBox(f'{num_rods} Zylinder wurden zufällig im 7x7x7 cm Würfel platziert.')

    except:
        if ui:
            ui.messageBox('Failed:\n{}'.format(traceback.format_exc()))
