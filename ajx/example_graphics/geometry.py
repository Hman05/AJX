import jax.numpy as jnp
from ajx.definitions import RigidBodyParameters
from panda3d.core import Quat


def normalized(*args):
    from panda3d.core import LVector3

    myVec = LVector3(*args)
    myVec.normalize()
    return myVec


def append_square(tris, vertex, normal, color, texcoord, x1, y1, z1, x2, y2, z2, c):
    start_id = vertex.getWriteRow()
    # make sure we draw the sqaure in the right plane
    if x1 != x2:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y1, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y2, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y2 - 1, 2 * z2 - 1))

    else:
        vertex.addData3(x1, y1, z1)
        vertex.addData3(x2, y2, z1)
        vertex.addData3(x2, y2, z2)
        vertex.addData3(x1, y1, z2)

        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z1 - 1))
        normal.addData3(normalized(2 * x2 - 1, 2 * y2 - 1, 2 * z2 - 1))
        normal.addData3(normalized(2 * x1 - 1, 2 * y1 - 1, 2 * z2 - 1))

    # adding different colors to the vertex for visibility
    color.addData4f(c[0], c[1], c[2], 1.0)
    color.addData4f(c[0], c[1], c[2], 1.0)
    color.addData4f(c[0], c[1], c[2], 1.0)
    color.addData4f(c[0], c[1], c[2], 1.0)

    texcoord.addData2f(0.0, 1.0)
    texcoord.addData2f(0.0, 0.0)
    texcoord.addData2f(1.0, 0.0)
    texcoord.addData2f(1.0, 1.0)

    tris.addVertices(start_id + 0, start_id + 1, start_id + 3)
    tris.addVertices(start_id + 1, start_id + 2, start_id + 3)


def makeSquare(game, x1, y1, z1, x2, y2, z2, c=[1.0, 1.0, 1.0], name=""):
    from panda3d.core import GeomVertexFormat, GeomVertexData
    from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
    from panda3d.core import GeomNode

    frmt = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData("square", frmt, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    color = GeomVertexWriter(vdata, "color")
    texcoord = GeomVertexWriter(vdata, "texcoord")
    tris = GeomTriangles(Geom.UHDynamic)

    append_square(tris, vertex, normal, color, texcoord, x1, y1, z1, x2, y2, z2, c)

    # Quads aren't directly supported by the Geom interface
    # you might be interested in the CardMaker class if you are
    # interested in rectangle though

    square = Geom(vdata)
    square.addPrimitive(tris)
    snode1 = GeomNode(name)
    snode1.addGeom(square)
    new = game.attachNewNode(snode1)
    new.setTwoSided(True)
    return new


class Box:
    def __init__(
        self,
        name,
        half_extent_x,
        half_extent_y,
        half_extent_z,
        translation=None,
        rotation=None,
        color=None,
    ):
        self.name = name
        self.half_extent_x = half_extent_x
        self.half_extent_y = half_extent_y
        self.half_extent_z = half_extent_z
        if translation:
            self.translation = translation
        else:
            self.translation = (0.0, 0.0, 0.0)
        self.rotation = rotation
        if color:
            self.color = color
        else:
            self.color = (0.3, 0.6, 0.3)
        self.node = None

    def get_diag_inertia(self, density):
        volume = self.half_extent_x * self.half_extent_y * self.half_extent_z * 8
        mass = volume * density
        J_x = mass / 3 * (self.half_extent_y**2 + self.half_extent_z**2)
        J_y = mass / 3 * (self.half_extent_x**2 + self.half_extent_z**2)
        J_z = mass / 3 * (self.half_extent_x**2 + self.half_extent_y**2)
        return jnp.array([J_x, J_y, J_z])

    def create_param(self, density, name):
        volume = self.half_extent_x * self.half_extent_y * self.half_extent_z * 8
        mass = volume * density
        return RigidBodyParameters.create(density, self.get_diag_inertia(mass), name)

    def create_node(self, game):
        from panda3d.core import GeomVertexFormat, GeomVertexData
        from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
        from panda3d.core import GeomNode

        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData("square", format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcolor = GeomVertexWriter(vdata, "color")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        tris = GeomTriangles(Geom.UHDynamic)

        x1 = -self.half_extent_x + self.translation[0]
        x2 = self.half_extent_x + self.translation[0]
        y1 = -self.half_extent_y + self.translation[1]
        y2 = self.half_extent_y + self.translation[1]
        z1 = -self.half_extent_z + self.translation[2]
        z2 = self.half_extent_z + self.translation[2]
        c = self.color

        append_square(
            tris, vertex, normal, texcolor, texcoord, x1, y1, z1, x2, y2, z1, c
        )
        append_square(
            tris, vertex, normal, texcolor, texcoord, x1, y1, z1, x2, y1, z2, c
        )
        append_square(
            tris, vertex, normal, texcolor, texcoord, x1, y1, z1, x1, y2, z2, c
        )
        append_square(
            tris, vertex, normal, texcolor, texcoord, x1, y1, z2, x2, y2, z2, c
        )
        append_square(
            tris, vertex, normal, texcolor, texcoord, x1, y2, z1, x2, y2, z2, c
        )
        append_square(
            tris, vertex, normal, texcolor, texcoord, x2, y1, z1, x2, y2, z2, c
        )

        cuboid = Geom(vdata)
        cuboid.addPrimitive(tris)
        snode1 = GeomNode(self.name)
        snode1.addGeom(cuboid)
        new = game.attachNewNode(snode1)
        new.setTwoSided(True)
        self.node = new

    def update_node(self, pos, rot):
        panda_rotation = Quat(*rot)
        self.node.setPos(*pos)
        self.node.setHpr(panda_rotation.getHpr())

    def inertia(self, mass):
        Jx = 1 / 12 * mass * (self.half_extent_y**2 + self.half_extent_z**2)
        Jy = 1 / 12 * mass * (self.half_extent_x**2 + self.half_extent_z**2)
        Jz = 1 / 12 * mass * (self.half_extent_y**2 + self.half_extent_x**2)
        return jnp.array([Jx, Jy, Jz])

    def extents_from_interia(inertia, mass):
        mat = jnp.ones([3, 3]) - jnp.eye(3)
        extents_squred = jnp.linalg.solve(12 * mass * mat, inertia)
        extents = jnp.sqrt(extents_squred)
        extents = jnp.nan_to_num(extents, nan=0.001)
        return extents[0], extents[1], extents[2]


class Square:
    def __init__(
        self,
        name,
        half_extent_x,
        half_extent_z,
        translation=None,
        rotation=None,
        color=None,
    ):
        self.name = name
        self.half_extent_x = half_extent_x
        self.half_extent_z = half_extent_z
        if translation:
            self.translation = translation
        else:
            self.translation = (0.0, 0.0, 0.0)
        self.rotation = rotation
        if color:
            self.color = color
        else:
            self.color = (0.3, 0.6, 0.3)
        self.node = None

    def create_node(self, game):
        from panda3d.core import GeomVertexFormat, GeomVertexData
        from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
        from panda3d.core import GeomNode

        format = GeomVertexFormat.getV3n3cpt2()
        vdata = GeomVertexData("square", format, Geom.UHDynamic)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcolor = GeomVertexWriter(vdata, "color")
        texcoord = GeomVertexWriter(vdata, "texcoord")
        tris = GeomTriangles(Geom.UHDynamic)

        x1 = -self.half_extent_x + self.translation[0]
        x2 = self.half_extent_x + self.translation[0]
        y1 = self.translation[1]
        y2 = self.translation[1]
        z1 = -self.half_extent_z + self.translation[2]
        z2 = self.half_extent_z + self.translation[2]

        append_square(
            tris, vertex, normal, texcolor, texcoord, x1, y1, z1, x2, y2, z2, self.color
        )

        # Quads aren't directly supported by the Geom interface
        # you might be interested in the CardMaker class if you are
        # interested in rectangle though

        square = Geom(vdata)
        square.addPrimitive(tris)
        snode1 = GeomNode(self.name)
        snode1.addGeom(square)
        new = game.attachNewNode(snode1)
        new.setTwoSided(True)
        self.node = new

    def update_node(self, pos, rot):
        panda_rotation = Quat(*rot)
        self.node.setPos(*pos)
        self.node.setHpr(panda_rotation.getHpr())


def makeCuboid(game, x1, y1, z1, x2, y2, z2, c=[1.0, 1.0, 1.0], name=""):
    from panda3d.core import GeomVertexFormat, GeomVertexData
    from panda3d.core import Geom, GeomTriangles, GeomVertexWriter
    from panda3d.core import GeomNode

    frmt = GeomVertexFormat.getV3n3cpt2()
    vdata = GeomVertexData("square", frmt, Geom.UHDynamic)

    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    color = GeomVertexWriter(vdata, "color")
    texcoord = GeomVertexWriter(vdata, "texcoord")
    tris = GeomTriangles(Geom.UHDynamic)

    append_square(tris, vertex, normal, color, texcoord, x1, y1, z1, x2, y2, z1, c)
    append_square(tris, vertex, normal, color, texcoord, x1, y1, z1, x2, y1, z2, c)
    append_square(tris, vertex, normal, color, texcoord, x1, y1, z1, x1, y2, z2, c)
    append_square(tris, vertex, normal, color, texcoord, x1, y1, z2, x2, y2, z2, c)
    append_square(tris, vertex, normal, color, texcoord, x1, y2, z1, x2, y2, z2, c)
    append_square(tris, vertex, normal, color, texcoord, x2, y1, z1, x2, y2, z2, c)

    cuboid = Geom(vdata)
    cuboid.addPrimitive(tris)
    snode1 = GeomNode(name)
    snode1.addGeom(cuboid)
    new = game.attachNewNode(snode1)
    new.setTwoSided(True)
    return new


class Model:
    def __init__(self, name, model_path, translation=None, rotation=None, scale=None):
        self.name = name
        if translation is not None:
            self.translation = translation
        else:
            self.translation = (0.0, 0.0, 0.0)
        if not rotation is None:
            if len(rotation) == 4:
                rotation = Quat(*rotation).getHpr()
            self.rotation = rotation
        else:
            self.rotation = (0.0, 0.0, 0.0)
        if scale is not None:
            self.scale = scale
        else:
            self.scale = (1.0, 1.0, 1.0)
        self.node = None
        self.model_path = model_path

    def create_node(self, game):
        self.node = render.attachNewNode("root")

        model = loader.loadModel(self.model_path)
        model.reparentTo(self.node)
        model.setPos(*self.translation)
        model.setHpr(*self.rotation)
        model.setScale(*self.scale)

    def update_node(self, pos, rot):
        self.node.setPos(*pos)
        self.node.setHpr(Quat(*rot).getHpr())
