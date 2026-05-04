import math
from abaqus import *
from abaqusConstants import *
import part, sketch, regionToolset


model_name = "MetaMaterialModule"

# Create model
if model_name in mdb.models:
    del mdb.models[model_name]

mdl = mdb.Model(name=model_name)


s = mdl.ConstrainedSketch(name='__profile__', sheetSize=max(w,h,d)*3)








# ============================================================
# PARAMETERS  (all dimensions in mm)
# ============================================================
TABLE_HEIGHT   = 720.0   # Overall height of the base
TABLE_WIDTH    = 600.0   # Width (distance between leg feet, side-to-side)
TABLE_DEPTH    = 600.0   # Depth (front to back, same as width for square base)

LEG_WIDTH      = 50.0    # Width of each diagonal leg (rectangular cross-section)
LEG_THICK      = 30.0    # Thickness of each diagonal leg

RAIL_HEIGHT    = 30.0    # Height of top / bottom horizontal rails
RAIL_THICK     = 30.0    # Thickness of rails (front-to-back)

NOTCH_DEPTH    = LEG_THICK / 2.0   # Half-lap notch depth where legs cross

BASE_RADIUS    = 80.0    # Radius of the circular foot pad
BASE_THICK     = 20.0    # Thickness of the foot pad

# ============================================================
# DERIVED GEOMETRY
# ============================================================
# The X-legs span from bottom corners to top rail, crossing in the middle.
# We model ONE pair of X-legs (viewed from the front) as two swept solids,
# then pattern / mirror for the second pair (viewed from the side).
#
# Leg centreline diagonal angle (from vertical):
half_w  = TABLE_WIDTH  / 2.0
half_h  = TABLE_HEIGHT / 2.0
leg_angle_rad = math.atan2(half_w, TABLE_HEIGHT)   # angle from vertical
leg_angle_deg = math.degrees(leg_angle_rad)

model_name = 'XBraceTableBase'
part_name  = 'TableBase'

# ============================================================
# HELPER: create a rectangular solid block
# ============================================================
def make_box(mdl, pname, w, h, d):
    """Return a Part that is a solid box (w x h x d)."""
    s = mdl.ConstrainedSketch(name='__profile__', sheetSize=max(w,h,d)*3)
    s.rectangle(point1=(0.0, 0.0), point2=(w, h))
    p = mdl.Part(name=pname, dimensionality=THREE_D,
                 type=DEFORMABLE_BODY)
    p.BaseSolidExtrude(sketch=s, depth=d)
    del mdl.sketches['__profile__']
    return p


def make_cylinder(mdl, pname, radius, height):
    """Return a Part that is a solid cylinder."""
    s = mdl.ConstrainedSketch(name='__profile__', sheetSize=radius*6)
    s.CircleByCenterPerimeter(center=(0.0, 0.0),
                              point1=(radius, 0.0))
    p = mdl.Part(name=pname, dimensionality=THREE_D,
                 type=DEFORMABLE_BODY)
    p.BaseSolidExtrude(sketch=s, depth=height)
    del mdl.sketches['__profile__']
    return p


# ============================================================
# CREATE MODEL
# ============================================================
if model_name in mdb.models:
    del mdb.models[model_name]

mdl = mdb.Model(name=model_name)

# ============================================================
# 1. DIAGONAL LEGS  (front X-frame, then mirrored for rear)
# ============================================================
# Each diagonal leg is a box whose length = hypotenuse of the frame.
leg_length = math.sqrt(TABLE_WIDTH**2 + TABLE_HEIGHT**2)

leg_front_L = make_box(mdl, 'Leg_Front_Left',  LEG_WIDTH, LEG_THICK, leg_length)
leg_front_R = make_box(mdl, 'Leg_Front_Right', LEG_WIDTH, LEG_THICK, leg_length)

# ============================================================
# 2. HORIZONTAL RAILS (top and bottom, running front-to-back)
# ============================================================
rail_top = make_box(mdl, 'Rail_Top',
                    TABLE_WIDTH, RAIL_HEIGHT, TABLE_DEPTH)
rail_bot = make_box(mdl, 'Rail_Bottom',
                    TABLE_WIDTH, RAIL_HEIGHT, TABLE_DEPTH)

# ============================================================
# 3. FOOT PAD (circular disc at base)
# ============================================================
foot_pad = make_cylinder(mdl, 'FootPad', BASE_RADIUS, BASE_THICK)

# ============================================================
# 4. ASSEMBLY
# ============================================================
a = mdl.rootAssembly
a.DatumCsysByDefault(CARTESIAN)

# --- Top rail instance ---
inst_rail_top = a.Instance(name='Rail_Top-1', part=rail_top, dependent=ON)
# Position: top of the frame, centred on X, Y = TABLE_HEIGHT - RAIL_HEIGHT
a.translate(instanceList=('Rail_Top-1',),
            vector=(-TABLE_WIDTH/2.0, TABLE_HEIGHT - RAIL_HEIGHT, -TABLE_DEPTH/2.0))

# --- Bottom rail instance ---
inst_rail_bot = a.Instance(name='Rail_Bottom-1', part=rail_bot, dependent=ON)
a.translate(instanceList=('Rail_Bottom-1',),
            vector=(-TABLE_WIDTH/2.0, 0.0, -TABLE_DEPTH/2.0))

# --- Foot pad (centred at origin, bottom of base) ---
inst_foot = a.Instance(name='FootPad-1', part=foot_pad, dependent=ON)
a.translate(instanceList=('FootPad-1',),
            vector=(-BASE_RADIUS, -BASE_THICK, -BASE_RADIUS))

# --- Front-left diagonal leg ---
# The leg box has its long axis along Z.  We rotate it to run diagonally:
#   * Rotate about the X-axis by +leg_angle_deg so it tilts in the Y-Z plane  — NO
#   * The X lies in the X-Y plane, so we rotate the leg about the Z-axis.
#
# Strategy:
#   1. Rotate -leg_angle_deg about Z (pivot at box origin = leg bottom-left corner).
#   2. Translate so the bottom of the leg sits at the bottom-left corner of the frame.

inst_fl = a.Instance(name='Leg_Front_Left-1', part=leg_front_L, dependent=ON)
# Rotate about Z axis (tilts the leg from bottom-left to top-right)
a.rotate(instanceList=('Leg_Front_Left-1',),
         axisPoint=(0.0, 0.0, 0.0),
         axisDirection=(0.0, 0.0, 1.0),
         angle=leg_angle_deg)
# Translate: bottom-left corner of frame is (-TABLE_WIDTH/2, 0, -TABLE_DEPTH/2)
a.translate(instanceList=('Leg_Front_Left-1',),
            vector=(-TABLE_WIDTH/2.0 - LEG_WIDTH/2.0,
                     0.0,
                    -TABLE_DEPTH/2.0))

# --- Front-right diagonal leg (mirror of front-left) ---
inst_fr = a.Instance(name='Leg_Front_Right-1', part=leg_front_R, dependent=ON)
# This leg goes from bottom-right to top-left, so rotate by -(leg_angle_deg)
a.rotate(instanceList=('Leg_Front_Right-1',),
         axisPoint=(0.0, 0.0, 0.0),
         axisDirection=(0.0, 0.0, 1.0),
         angle=-leg_angle_deg)
# Translate: bottom-right corner
a.translate(instanceList=('Leg_Front_Right-1',),
            vector=(TABLE_WIDTH/2.0 - LEG_WIDTH/2.0,
                    0.0,
                   -TABLE_DEPTH/2.0))

# --- Rear X-frame: linear pattern of both front legs along depth axis ---
a.LinearInstancePattern(
    instanceList=('Leg_Front_Left-1', 'Leg_Front_Right-1'),
    direction1=(0.0, 0.0, 1.0),
    direction2=(1.0, 0.0, 0.0),
    number1=2,
    number2=1,
    spacing1=TABLE_DEPTH,
    spacing2=1.0)

# ============================================================
# 5. MERGE INTO A SINGLE PART (optional but recommended)
# ============================================================
# Collect all instances
all_instances = tuple(a.instances.keys())
a.InstanceFromBooleanMerge(
    name=part_name,
    instances=[a.instances[k] for k in all_instances],
    originalInstances=DELETE,
    domain=GEOMETRY)

# ============================================================
# 6. VIEWPORT & SAVE
# ============================================================
session.viewports['Viewport: 1'].setValues(displayedObject=a)
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
session.viewports['Viewport: 1'].view.fitView()

mdb.saveAs(pathName='x_brace_table_base.cae')

print('='*60)
print('X-Brace Table Base model created successfully.')
print('  Model  : %s' % model_name)
print('  Part   : %s' % part_name)
print('  Height : %.1f mm' % TABLE_HEIGHT)
print('  Width  : %.1f mm' % TABLE_WIDTH)
print('  Depth  : %.1f mm' % TABLE_DEPTH)
print('Saved as : x_brace_table_base.cae')
print('='*60)