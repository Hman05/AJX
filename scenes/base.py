import jax
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *

import jax.numpy as jnp
import numpy as np
from jax import jit
from graphics.geometry import Box, Square, Model
from graphics.default_scene import Game
from environments.pendulum import Pendulum
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
from util.deepinsert import deepinsert
from loguru import logger


class GraphicalEnvironmentBase(DirectObject):
    def __init__(
        self,
        env,
        env_param,
        initial_state,
        trajectory_path=None,
        synchronized_start_k=0,
    ):
        self.timestep = env.timestep
        self.environment = env
        self.env_param = env_param
        self.initial_state = initial_state
        self.game = Game(framerate=1 / self.timestep)

        self.force = env.force
        self.step = jit(env.step)
        self.observe = jit(env.observe)

        self.physics_is_active = True
        self.replay_active = False

        self.state = copy.deepcopy(self.initial_state)
        self.observation = jnp.zeros([len(env.observable_names)])

        self.create_graphics(initial_state)
        self.create_events()

        self.u = jnp.array([0.0])
        self.last_observation = None
        self.contoller_is_active = True
        self.reset_k = synchronized_start_k
        self.k = self.reset_k
        self.counter = 0

        self.state_sequence = None
        if trajectory_path:
            import pickle

            with open(trajectory_path, "rb") as input_file:
                raw_data = pickle.load(input_file)

            if type(raw_data).__name__ == "TrajectoryDataset":
                self.state_sequence = raw_data.initial_states
            else:
                self.state_sequence = env.xarray_to_trajectory(
                    raw_data["trajectory"].states
                )

    def create_graphics(self, initial_state):
        self.geometry_dict = {
            geometry.name: geometry for geometry in self.environment.geometry_list
        }
        self.hidden_geometry_dict = dict()
        for rb in self.environment.rigid_bodies:
            for g_name in rb.geometry:
                assert (
                    g_name in self.geometry_dict
                ), f"Unknown geometry label found. Did you forget to add {g_name} to the geometry list?"
                geometry = self.geometry_dict[g_name]
                geometry.create_node(self.game)
        render.flattenLight()

        if jax.default_backend() == "gpu":
            logger.warning("Running physics on GPU. CPU is likely faster.")

        for geometry in self.environment.extra_geometry:
            geometry.create_node(self.game)

        self.text_toggle_physics = OnscreenText(
            text="p: Toggle physics",
            style=1,
            fg=(1, 1, 1, 1),
            pos=(0.06, -0.08),
            align=TextNode.ALeft,
            scale=0.05,
            parent=base.a2dTopLeft,
        )

        self.text_toggle_controller = OnscreenText(
            text="c: Toggle controller",
            style=1,
            fg=(1, 1, 1, 1),
            pos=(0.06, -0.14),
            align=TextNode.ALeft,
            scale=0.05,
            parent=base.a2dTopLeft,
        )

        self.text_toggle_playback = OnscreenText(
            text="t: Toggle playback",
            style=1,
            fg=(1, 1, 1, 1),
            pos=(0.06, -0.20),
            align=TextNode.ALeft,
            scale=0.05,
            parent=base.a2dTopLeft,
        )
        self.text_display_observations1 = OnscreenText(
            text="",
            style=1,
            fg=(1, 1, 1, 1),
            pos=(0.06, -0.26),
            align=TextNode.ALeft,
            scale=0.05,
            parent=base.a2dTopLeft,
        )
        self.text_display_observations2 = OnscreenText(
            text="",
            style=1,
            fg=(1, 1, 1, 1),
            pos=(0.06, -0.32),
            align=TextNode.ALeft,
            scale=0.05,
            parent=base.a2dTopLeft,
        )
        self.text_display_observations3 = OnscreenText(
            text="",
            style=1,
            fg=(1, 1, 1, 1),
            pos=(0.06, -0.38),
            align=TextNode.ALeft,
            scale=0.05,
            parent=base.a2dTopLeft,
        )
        base.camLens.setNearFar(0.01, 200)

        # Load the skybox
        # self.skybox = loader.loadModel("skybox2.bam")
        # self.skybox.setScale(200)
        # self.skybox.reparentTo(render)
        # self.skybox.setShaderOff()
        # self.skybox.setBin("background", 0)
        # self.skybox.setDepthWrite(0)
        # self.skybox.setLightOff()
        # self.skybox.setTwoSided(True)

        # define the colors at the top ("sky"), bottom ("ground") and center
        # ("horizon") of the background gradient
        sky_color = (1.0, 1.0, 1.0, 1.0)
        horizon_color = (0.1, 0, 0.8, 1.0)  # optional
        ground_color = (0, 0.0, 0.2, 1.0)
        # sky_color = (1.0, 1.0, 1.0, 1.0)
        # horizon_color = (1.0, 1.0, 1.0, 1.0)
        # ground_color = (1.0, 1.0, 1.0, 1.0)
        self.background_gradient = self.create_gradient(
            sky_color, ground_color, horizon_color
        )
        # looks like the background needs to be parented to an intermediary node
        # to which a compass effect is applied to keep it at the same position
        # as the camera, while being parented to render
        pivot = render.attach_new_node("pivot")
        effect = CompassEffect.make(camera, CompassEffect.P_pos)
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
                1,  # left triangle; should never be in view
                3,
                4,
                5,  # right triangle; should never be in view
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

    def create_events(self):
        self.key_map = {
            "l": False,
            "h": False,
            "j": False,
            "k": False,
            "u": False,
            "i": False,
            "m": False,
            ",": False,
            "n": False,
            "y": False,
        }
        self.game.setStepFunction(self.update)

        def toggle_physics():
            self.physics_is_active = not self.physics_is_active

        def toggle_replay():
            self.replay_active = not self.replay_active

        def toggle_controller():
            self.contoller_is_active = not self.contoller_is_active

        def update_key_map(key, state):
            self.key_map[key] = state

        self.game.setResetFunction(self.reset)
        self.accept("p", toggle_physics)
        self.accept("c", toggle_controller)
        self.accept("t", toggle_replay)

        self.accept("l", update_key_map, ["l", True])
        self.accept("l-up", update_key_map, ["l", False])
        self.accept("h", update_key_map, ["h", True])
        self.accept("h-up", update_key_map, ["h", False])
        self.accept("j", update_key_map, ["j", True])
        self.accept("j-up", update_key_map, ["j", False])
        self.accept("k", update_key_map, ["k", True])
        self.accept("k-up", update_key_map, ["k", False])
        self.accept("u", update_key_map, ["u", True])
        self.accept("u-up", update_key_map, ["u", False])
        self.accept("i", update_key_map, ["i", True])
        self.accept("i-up", update_key_map, ["i", False])
        self.accept("m", update_key_map, ["m", True])
        self.accept("m-up", update_key_map, ["m", False])
        self.accept(",", update_key_map, [",", True])
        self.accept(",-up", update_key_map, [",", False])
        self.accept("n", update_key_map, ["n", True])
        self.accept("n-up", update_key_map, ["n", False])
        self.accept("y", update_key_map, ["y", True])
        self.accept("y-up", update_key_map, ["y", False])

    def reset(self):
        if self.replay_active:
            self.k = self.reset_k
        else:
            self.state = self.initial_state

    def update(self, task):
        self.pre_update(task)
        simulate_physics = self.physics_is_active and (not self.replay_active)
        if simulate_physics:
            self.update_physics()
        elif self.replay_active:
            self.update_trajectory()
            self.observation = self.observe(self.state, None, self.env_param)

        info_list = self.environment.get_display_text(self.observation)
        if len(info_list) > 0:
            self.text_display_observations1.setText(info_list[0])
        if len(info_list) > 1:
            self.text_display_observations2.setText(info_list[1])
        if len(info_list) > 2:
            self.text_display_observations3.setText(info_list[2])
        # Very slow...
        sim = 2
        r = (self.counter * sim) % len(self.environment.rigid_bodies)
        for i, rb in enumerate(self.environment.rigid_bodies):  # [r : r + sim]
            i = i
            for g_name in rb.geometry:
                # assert g_name in self.geometry_dict
                geometry = self.geometry_dict[g_name]
                geometry.update_node(self.state.conf.pos[i], self.state.conf.rot[i])

        if self.physics_is_active and self.replay_active:
            self.k += 1
        self.counter += 1

        return task.cont

    def pre_update(self, task):
        self.u = self.environment.control_func(
            self.observation, self.last_observation, self.key_map
        )

    def update_physics(self):
        qdot_next, ghosts, status = self.force(self.state, -self.u, self.env_param)
        self.last_observation = self.observation
        self.observation = self.observe(self.state, qdot_next, self.env_param)
        self.state = self.step(self.state, qdot_next)

        assert status == 0

    def update_trajectory(self):
        if not self.state_sequence is None:
            traj_length = self.state_sequence.conf.pos.shape[0]
            self.state = self.state_sequence[self.k % traj_length]

    def run(self):
        self.game.enableMouse()
        self.game.run()


if __name__ == "__main__":
    timestep = 0.016667
    environment = Pendulum(
        override_param={},
        timestep=timestep,
        has_dry_friction=False,
        has_quadratic_damping=False,
        has_stribeck_effect=False,
        has_inverse_compliance=False,
    )
    theta = 3.0
    initial_state = environment.state_from_angle(theta, environment.param)

    controller = GraphicalEnvironmentBase(environment, initial_state)

    controller.run()
