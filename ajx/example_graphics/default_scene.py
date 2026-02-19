import sys
from direct.showbase.ShowBase import ShowBase
from direct.showbase.DirectObject import DirectObject
from direct.gui.DirectGui import *
from direct.interval.IntervalGlobal import *
from direct.showbase.ShowBaseGlobal import globalClock
from direct.showbase.InputStateGlobal import inputState

from panda3d.core import AmbientLight
from panda3d.core import DirectionalLight
from panda3d.core import LVector3
from panda3d.core import PointLight
from panda3d.core import ClockObject


# You can't normalize inline so this is a helper function
def normalized(*args):
    myVec = LVector3(*args)
    myVec.normalize()
    return myVec


class Game(ShowBase):
    def __init__(self, framerate=60):
        ShowBase.__init__(self)
        base.set_background_color(0.1, 0.1, 0.8, 1)
        base.set_frame_rate_meter(True)

        # base.cam.set_pos(0, -20, 4)
        # base.cam.look_at(0, 0, 0)

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
        self.accept("f1", base.toggle_wireframe)
        self.accept("f2", base.toggle_texture)

        inputState.watchWithModifiers("forward", "w")
        inputState.watchWithModifiers("left", "a")
        inputState.watchWithModifiers("reverse", "s")
        inputState.watchWithModifiers("right", "d")
        inputState.watchWithModifiers("turnLeft", "q")
        inputState.watchWithModifiers("turnRight", "e")

        # Physics
        # self.setup()
        self.step_func = lambda self: None
        self.updateTask = taskMgr.add(self.step_func, "update")

        clock = ClockObject.getGlobalClock()
        clock.setMode(ClockObject.MLimited)
        clock.setFrameRate(framerate)

    def do_exit(self):
        sys.exit(0)

    def do_reset(self):
        pass

    def process_input(self, dt):
        force = LVector3(0, 0, 0)
        torque = LVector3(0, 0, 0)

        if inputState.isSet("forward"):
            force.y = 1.0
        if inputState.isSet("reverse"):
            force.y = -1.0
        if inputState.isSet("left"):
            force.x = -1.0
        if inputState.isSet("right"):
            force.x = 1.0
        if inputState.isSet("turnLeft"):
            torque.z = 1.0
        if inputState.isSet("turnRight"):
            torque.z = -1.0

        force *= 30.0
        torque *= 10.0

        force = render.get_relate()

    def attachNewNode(self, node):
        return render.attachNewNode(node)

    def setStepFunction(self, step_func):
        self.step_func = step_func
        self.updateTask = taskMgr.add(self.step_func, "update")

    def setResetFunction(self, reset_func):
        self.reset_func = reset_func
        self.updateTask = self.accept("r", reset_func)


def run():
    base = Game()
    base.run()
