"""Geometric locus of an EEF when only one revolute joint moves."""
import numpy as np


def eef_circle(joint_origin, joint_axis, eef, samples=181):
    """Project EEF onto the axis for the centre; rotate the radial vector by 2*pi."""
    origin,axis,p = map(lambda x: np.asarray(x,dtype=float),(joint_origin,joint_axis,eef))
    axis = axis/np.linalg.norm(axis)
    centre = origin+axis*np.dot(p-origin,axis)
    radial = p-centre
    if np.linalg.norm(radial)<1e-6:
        return p[None,:]  # On the axis: the circle degenerates to a point.
    angles = np.linspace(0,2*np.pi,samples)
    return centre+np.cos(angles)[:,None]*radial+np.sin(angles)[:,None]*np.cross(axis,radial)
