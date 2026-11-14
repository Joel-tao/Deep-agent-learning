"""Candidate geometry and aperture shaping; not a certified grasp condition.

Offsets follow the scripted controller's cube-centre +2mm EEF target.
Tolerance scales (30mm lateral, 25mm vertical, 70mm slider separation) are
initial heuristic settings, not tuned optima or exact finger-pad clearance.
"""
import numpy as np

PROFILE='prepare_v1'
PARAMETERS=dict(horizontal_scale_m=.03,vertical_scale_m=.025,
                eef_height_offset_m=.002,open_slider_separation_m=.07,
                close_bonus_budget=.1)


def preparation_terms(eef,cube,slider_separation,close_command,contact,hold_progress,reach):
    delta=np.asarray(eef)-np.asarray(cube)
    xy=float(np.linalg.norm(delta[:2]))
    height=float(delta[2]-.002)
    readiness=float(np.exp(-((xy/.03)**2+(height/.025)**2)))
    opening=float(np.clip(slider_separation/.07,0,1))
    closing=float(np.clip(close_command,0,1))
    # Preserve approach information but favor open approach before contact.
    prepared_reach=reach*(.25+.75*opening) if not contact else reach
    # Merely requesting closure earns at most 0.1 of the component. Actual
    # bilateral contact always exceeds that cap, then sustained contact ramps up.
    grasp=(.1+.9*hold_progress) if contact else .1*readiness*closing
    return dict(reach=float(prepared_reach),grasp=float(grasp)),dict(
        readiness=readiness,horizontal_error=xy,height_error=height,
        opening_progress=opening,premature_close=bool(closing>.1 and readiness<.2))
