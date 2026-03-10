import jax
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *

import jax.numpy as jnp
import numpy as np
from jax import jit
from ajx.example_graphics.geometry import Box, Square, Model
import copy

from panda3d.core import (
    GeomVertexFormat,
    GeomVertexArrayFormat,
    InternalName,
    Geom,
    GeomVertexData,
    GeomEnums,
    GeomTriangles,
    GeomNode,
    OmniBoundingVolume,
    NodePath,
    CompassEffect,
    BillboardEffect,
    Vec3,
    Point3,
)
from direct.showbase.ShowBase import ShowBase
import array
from panda3d.core import Quat
from panda3d.core import TextNode
import scipy
from loguru import logger

from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import PointLight
from typing import Tuple


class EnvironmentScene:
    def __init__(
        self,
        env,
        env_param,
        initial_state,
    ):
        self.timestep = env.sim.settings.timestep
        self.environment = env
        self.env_param = env_param
        self.initial_state = initial_state

        self.step = jit(env.step)
        self.observe = jit(env.observe_state)

        self.physics_is_active = True
        self.displayed_information = 0

        self.state = copy.deepcopy(self.initial_state)
        self.observation = jnp.zeros([len(env.observable_names)])

        self.u = jnp.array([0.0])
        self.last_observation = None
        self.contoller_is_active = True

        self.state_sequence = None

    def setup(self, base: DirectObject, render):
        def toggle_physics():
            self.physics_is_active = not self.physics_is_active

        def cycle_information():
            self.displayed_information = (self.displayed_information + 1) % 4

        # Key bindings
        base.accept("p", toggle_physics)
        base.accept("o", cycle_information)

        # Light
        alight = AmbientLight("ambientLight")
        alight.set_color((0.5, 0.5, 0.5, 1))
        alightNP = render.attach_new_node(alight)

        dlight = DirectionalLight("directionalLight")
        dlight.set_direction((1, 1, -1))
        dlight.set_color((0.7, 0.7, 0.7, 1))
        dlightNP = render.attach_new_node(dlight)

        plight = PointLight("plight")
        plight.set_color((0.0, 1.0, 0.0, 1))
        plight.attenuation = (1, 0, 1)
        plnp = render.attachNewNode(plight)
        plnp.setPos(0, 0.5, 0)

        render.clear_light()
        render.set_light(alightNP)
        render.set_light(dlightNP)

        self.geometry_dict = {
            geometry.name: geometry for geometry in self.environment.geometry_list
        }
        self.hidden_geometry_dict = dict()
        for rb in self.environment.sim.rigid_body_list:
            for g_name in rb.geometry:
                assert (
                    g_name in self.geometry_dict
                ), f"Unknown geometry label found. Did you forget to add {g_name} to the geometry list?"
                geometry = self.geometry_dict[g_name]
                geometry.create_node(base)
        render.flattenLight()

        if jax.default_backend() == "gpu":
            logger.warning("Running physics on GPU. CPU is likely faster.")

        for geometry in self.environment.extra_geometry:
            geometry.create_node(base)

        self.text_displays = [
            OnscreenText(
                text="",
                style=1,
                fg=(1, 1, 1, 1),
                pos=(0.06, -0.08 - 0.06 * i),
                align=TextNode.ALeft,
                scale=0.05,
                parent=base.a2dTopLeft,
            )
            for i in range(10)
        ]
        base.camLens.setNearFar(0.01, 2000)

        # Create sky gradient
        sky_color = (1.0, 1.0, 1.0, 1.0)
        horizon_color = (0.1, 0, 0.8, 1.0)  # optional
        ground_color = (0, 0.0, 0.2, 1.0)
        self.background_gradient = self.create_gradient(
            sky_color, ground_color, horizon_color
        )
        pivot = render.attach_new_node("pivot")
        effect = CompassEffect.make(base.camera, CompassEffect.P_pos)
        pivot.set_effect(effect)
        self.background_gradient.reparent_to(pivot)
        # now the background model just needs to keep facing the camera (only
        # its heading should correspond to that of the camera; its pitch and
        # roll need to remain unaffected)
        effect = BillboardEffect.make(
            Vec3.up(),
            False,
            True,
            0.0,
            NodePath(),
            # make the background model face a point behind the camera
            Point3(0.0, -10.0, 0.0),
            False,
        )
        self.background_gradient.set_effect(effect)

    def create_gradient(self, sky_color, ground_color, horizon_color=None):
        vertex_format = GeomVertexFormat()
        array_format = GeomVertexArrayFormat()
        array_format.add_column(
            InternalName.get_vertex(), 3, Geom.NT_float32, Geom.C_point
        )
        vertex_format.add_array(array_format)
        array_format = GeomVertexArrayFormat()
        array_format.add_column(
            InternalName.make("color"), 4, Geom.NT_uint8, Geom.C_color
        )
        vertex_format.add_array(array_format)
        vertex_format = GeomVertexFormat.register_format(vertex_format)

        vertex_data = GeomVertexData("prism_data", vertex_format, GeomEnums.UH_static)
        vertex_data.unclean_set_num_rows(6)
        # create a simple, horizontal prism;
        # make it very wide to avoid ever seeing its left and right sides;
        # one edge is at the "horizon", while the two other edges are above
        # and a bit behind the camera so they are only visible when looking
        # straight up
        values = array.array(
            "f",
            [
                -100.0,
                -50.0,
                86.6,
                -100.0,
                100.0,
                0.0,
                -100.0,
                -50.0,
                -86.6,
                100.0,
                -50.0,
                86.6,
                100.0,
                100.0,
                0.0,
                100.0,
                -50.0,
                -86.6,
            ],
        )
        pos_array = vertex_data.modify_array(0)
        memview = memoryview(pos_array).cast("B").cast("f")
        memview[:] = values

        color1 = tuple(int(round(c * 255)) for c in sky_color)
        color3 = tuple(int(round(c * 255)) for c in ground_color)

        if horizon_color is None:
            color2 = tuple((c1 + c2) // 2 for c1, c2 in zip(color1, color3))
        else:
            color2 = tuple(int(round(c * 255)) for c in horizon_color)

        values = array.array("B", (color1 + color2 + color3) * 2)
        color_array = vertex_data.modify_array(1)
        memview = memoryview(color_array).cast("B")
        memview[:] = values

        tris_prim = GeomTriangles(GeomEnums.UH_static)
        indices = array.array(
            "H",
            [
                0,
                2,
                1,
                3,
                4,
                5,
                0,
                4,
                3,
                0,
                1,
                4,
                1,
                5,
                4,
                1,
                2,
                5,
                2,
                3,
                5,
                2,
                0,
                3,
            ],
        )
        tris_array = tris_prim.modify_vertices()
        tris_array.unclean_set_num_rows(24)
        memview = memoryview(tris_array).cast("B").cast("H")
        memview[:] = indices

        geom = Geom(vertex_data)
        geom.add_primitive(tris_prim)
        node = GeomNode("prism")
        node.add_geom(geom)
        # the compass effect can make the node leave its bounds, so make them
        # infinitely large
        node.set_bounds(OmniBoundingVolume())
        prism = NodePath(node)
        prism.set_light_off()
        prism.set_bin("background", 0)
        prism.set_depth_write(False)
        prism.set_depth_test(False)

        return prism

    def get_initial_camera_transform(self) -> Tuple[Quat, Vec3]:
        camera_rot = Quat(1.0, 0.0, 0.0, 0.0)
        camera_pos = Vec3(0.0, 0.0, 0.0)
        if hasattr(self.environment, "camera_rot"):
            camera_rot = Quat(*self.environment.camera_rot)
        if hasattr(self.environment, "camera_pos"):
            camera_pos = Vec3(*self.environment.camera_pos)

        return camera_pos, camera_rot

    def reset(self):
        self.state = self.initial_state
        self.observation = self.observe(self.state, -self.u, self.env_param)

    def update(self, key_map):
        self.pre_update(key_map)
        if self.physics_is_active:
            self.update_physics()
            self.update_geometry()

    def update_geometry(self):
        info_list = [
            "o: Cycle displayed information",
            "r: Restart scene",
            "p: Toggle physics",
        ]
        if self.displayed_information == 1:
            info_list = [
                "Left mouse + drag: Pan the camera",
                "Middle mouse + drag: Rotate the view",
                "Right mouse + drag: Zoom in/out",
            ]
        if self.displayed_information == 2:
            info_list = self.environment.control_help_strings()
        if self.displayed_information == 3:
            info_list = self.environment.observation_strings(self.observation)

        if len(info_list) > 0:
            for i in range(0, len(self.text_displays)):
                if len(info_list) > i:
                    self.text_displays[i].setText(info_list[i])
                else:
                    self.text_displays[i].setText("")

        rb_list = self.environment.sim.rigid_body_list
        geo_dict = self.geometry_dict

        # per timestep/frame:
        pos_np = np.asarray(jax.device_get(self.state.conf.pos))  # (N,3)
        rot_np = np.asarray(jax.device_get(self.state.conf.rot))  # (N,4)

        for i, rb in enumerate(rb_list):
            pos = pos_np[i]
            rot = rot_np[i]
            q = Quat(*rot)
            for g_name in rb.geometry:
                geo_dict[g_name].node.setPosQuat(Vec3(*pos), q)

    def pre_update(self, key_map):
        self.u = self.environment.control_func(
            self.observation, self.last_observation, key_map
        )

    def update_physics(self):
        self.last_observation = self.observation
        self.state, self.observation = self.step(self.state, -self.u, self.env_param)

    def update_trajectory(self):
        if not self.state_sequence is None:
            traj_length = self.state_sequence.conf.pos.shape[0]
            self.state = self.state_sequence[self.k % traj_length]

    def run(self):
        # self.game.enableMouse()
        self.game.run()
