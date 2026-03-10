import sys
from direct.showbase.ShowBase import ShowBase

ShowBaseGlobal = False
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.showbase.ShowBaseGlobal import globalClock
from direct.showbase.InputStateGlobal import inputState

from panda3d.core import LVector3
from panda3d.core import ClockObject
from panda3d.core import WindowProperties
from panda3d.core import Vec3, Quat
from panda3d.core import Trackball


class Application(ShowBase):
    def __init__(self, scene, framerate: int = 60, camera_controller: str = "fps"):
        ShowBase.__init__(self)
        self.scene = scene
        self.camera_controller = camera_controller
        self.create_events()
        self.init_mouse()
        self.reset_camera()

        self.set_background_color(0.1, 0.1, 0.8, 1)
        self.set_frame_rate_meter(True)

        scene.setup(self, self.render)

        self.accept("f1", self.toggle_wireframe)
        self.accept("f2", self.toggle_texture)

        self.updateTask = self.taskMgr.add(self.update, "update")
        self.resetTask = self.accept("r", self.reset)

        clock = ClockObject.getGlobalClock()
        clock.setMode(ClockObject.MLimited)
        clock.setFrameRate(framerate)

    def init_mouse(self):
        if self.camera_controller == "default":
            return
        self.disableMouse()
        if self.camera_controller == "fps":
            # Camera state
            self.cam_speed = 10.0
            self.mouse_sensitivity = 0.15
            self.pitch = 0.0
            self.yaw = 0.0

            # Lock / hide cursor for mouselook
            props = WindowProperties()
            props.setCursorHidden(True)
            self.win.requestProperties(props)

        self.camera_cooldown = 10

    def create_events(self):
        keys = list("qwertyuiopasdfghjklzxcvnm,.1234567890")
        keys.extend(["space", "shift"])
        # keys.extend(["mouse1", "mouse2", "mouse3"])
        self.key_map = {k: False for k in keys}

        def update_key_map(key, state):
            self.key_map[key] = state

        for k in keys:
            self.accept(k, update_key_map, [k, True])
            self.accept(f"{k}-up", update_key_map, [k, False])

    def reset(self):
        self.scene.reset()
        # self.reset_camera()

    def reset_camera(self):
        camera_pos, camera_rot = self.scene.get_initial_camera_transform()
        if self.camera_controller == "default":
            self.mouseInterfaceNode.setPos(camera_pos)
            self.mouseInterfaceNode.setHpr(camera_rot.getHpr())
            hpr = camera_rot.getHpr()
            self.yaw = hpr.x
            self.pitch = hpr.y

        # Starting position
        self.camera.setPos(camera_pos)
        self.camera.setQuat(camera_rot)
        hpr = camera_rot.getHpr()
        self.yaw = hpr.x
        self.pitch = hpr.y

    def update(self, task):
        self.update_camera()
        self.scene.update(self.key_map)
        return task.cont

    def update_camera(self):
        if self.camera_controller == "default":
            return
        if self.camera_controller == "fps":
            self.update_fps_camera()
        else:
            raise Exception

    def update_fps_camera(self):
        dt = globalClock.getDt()
        win = self.win
        cx = win.getXSize() // 2
        cy = win.getYSize() // 2

        if self.camera_cooldown > 0:
            win.movePointer(0, cx, cy)
            self.camera_cooldown -= 1
            return

        # Mouse look
        if self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            x = md.getX()
            y = md.getY()

            dx = x - cx
            dy = y - cy

            self.yaw -= dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = max(-89, min(89, self.pitch))
            q = q = Quat()
            q.setHpr(Vec3(self.yaw, self.pitch, 0))

            self.camera.setQuat(q)

            # recenter for next frame
            win.movePointer(0, cx, cy)

        # Movement relative to camera orientation
        move = Vec3(0, 0, 0)

        if self.key_map["w"]:
            move.y += 1.0
        if self.key_map["s"]:
            move.y -= 1.0
        if self.key_map["a"]:
            move.x -= 1.0
        if self.key_map["d"]:
            move.x += 1.0
        if self.key_map["space"]:
            move.z += 1.0
        if self.key_map["shift"]:
            move.z -= 1.0

        if move.length_squared() > 0:
            self.camera.setPos(self.camera, move * self.cam_speed * dt)

    def attachNewNode(self, node):
        return self.render.attachNewNode(node)


def run():
    base = Application()
    base.run()
