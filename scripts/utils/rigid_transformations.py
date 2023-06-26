"""
Lean rigid transformation class
Author: wangdexin
参考 dex-net
"""
import os
import numpy as np
from . import transformations


class RigidTransform(object):
    """A Rigid Transformation from one frame to another.
    """

    def __init__(self, rotation=np.eye(3), translation=np.zeros(3), from_frame='unassigned', to_frame='world'):
        """Initialize a RigidTransform.

        旋转矩阵和平移向量表示 从from_frame坐标系到to_frame坐标系的转换

        Parameters
        ----------
        rotation : :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix (should be unitary).

        translation : :obj:`numpy.ndarray` of float
            A 3-entry translation vector.

        from_frame : :obj:`str`
        to_frame : :obj:`str`

        """
        if not isinstance(from_frame, str):
            raise ValueError('Must provide string name of input frame of data')
        if not isinstance(to_frame, str):
            raise ValueError('Must provide string name of output frame of data')

        self.rotation = rotation
        self.translation = translation
        self._from_frame = from_frame
        self._to_frame = to_frame

    def copy(self):
        """Returns a copy of the RigidTransform.

        Returns
        -------
        :obj:`RigidTransform`
            A deep copy of the RigidTransform.
        """
        return RigidTransform(np.copy(self.rotation), np.copy(self.translation), self.from_frame, self.to_frame)

    def _check_valid_rotation(self, rotation):
        """Checks that the given rotation matrix is valid.
        """
        if not isinstance(rotation, np.ndarray) or not np.issubdtype(rotation.dtype, np.number):
            raise ValueError('Rotation must be specified as numeric numpy array')

        if len(rotation.shape) != 2 or rotation.shape[0] != 3 or rotation.shape[1] != 3:
            raise ValueError('Rotation must be specified as a 3x3 ndarray')

        if np.abs(np.linalg.det(rotation) - 1.0) > 1e-3:
            raise ValueError('Illegal rotation. Must have determinant == 1.0')

    def _check_valid_translation(self, translation):
        """Checks that the translation vector is valid.
        """
        if not isinstance(translation, np.ndarray) or not np.issubdtype(translation.dtype, np.number):
            raise ValueError('Translation must be specified as numeric numpy array')

        t = translation.squeeze()
        if len(t.shape) != 1 or t.shape[0] != 3:
            raise ValueError('Translation must be specified as a 3-vector, 3x1 ndarray, or 1x3 ndarray')

    @property
    def rotation(self):
        """:obj:`numpy.ndarray` of float: A 3x3 rotation matrix.
        """
        return self._rotation

    @rotation.setter
    def rotation(self, rotation):
        """
        rotation：
            四元数wxyz或旋转矩阵
        """
        # Convert quaternions
        if len(rotation) == 4:
            q = np.array([q for q in rotation])
            if np.abs(np.linalg.norm(q) - 1.0) > 1e-3:
                raise ValueError('Invalid quaternion. Must be norm 1.0')
            rotation = RigidTransform.rotation_from_quaternion(q)

        # Convert lists and tuples
        if type(rotation) in (list, tuple):
            rotation = np.array(rotation).astype(np.float32)

        self._check_valid_rotation(rotation)
        self._rotation = rotation * 1.

    @property
    def translation(self):
        """:obj:`numpy.ndarray` of float: A 3-ndarray that represents the
        transform's translation vector.
        """
        return self._translation

    @translation.setter
    def translation(self, translation):
        # Convert lists to translation arrays
        if type(translation) in (list, tuple) and len(translation) == 3:
            translation = np.array([t for t in translation]).astype(np.float32)

        self._check_valid_translation(translation)
        self._translation = translation.squeeze() * 1.
        
    @property
    def from_frame(self):
        """:obj:`str`: The identifier for the 'from' frame of reference.
        """
        return self._from_frame

    @from_frame.setter
    def from_frame(self, from_frame):
        self._from_frame = from_frame

    @property
    def to_frame(self):
        """:obj:`str`: The identifier for the 'to' frame of reference.
        """
        return self._to_frame

    @to_frame.setter
    def to_frame(self, to_frame):
        self._to_frame = to_frame

    @property
    def euler_angles(self):
        """:obj:`tuple` of float: The three euler angles for the rotation.
        """
        q_wxyz = self.quaternion
        q_xyzw = np.roll(q_wxyz, -1)
        return transformations.euler_from_quaternion(q_xyzw)

    @property
    def quaternion(self):
        """:obj:`numpy.ndarray` of float: A quaternion vector in wxyz layout.
        """
        q_xyzw = transformations.quaternion_from_matrix(self.matrix)
        q_wxyz = np.roll(q_xyzw, 1)
        return q_wxyz
    
    @property
    def quaternion_xyzw(self):
        """:obj:`numpy.ndarray` of float: A quaternion vector in xyzw layout.
        """
        q_xyzw = transformations.quaternion_from_matrix(self.matrix)
        return q_xyzw
    
    @property
    def euler(self):
        """TODO DEPRECATE THIS?"""
        e_xyz = transformations.euler_from_matrix(self.rotation, 'sxyz')
        return np.array([180.0 / np.pi * a for a in e_xyz])
    
    def transfor_points_from(self, points):
        """
        将to_frame 下的输入点 转换到 from_frame 下

        points: np.ndarray shape=(n, 3)
        """
        points = points.T  # 转置  (3, n)
        ones = np.ones((1, points.shape[1]))
        points = np.vstack((points, ones))  # (4, n)
        # 转换
        new_points = np.matmul(self.matrix, points)[:-1, :]   # (3, n)
        return new_points.T  # 转置  (n, 3)
    
    def transfor_points_to(self, points):
        """
        将from_frame 下的输入点 转换到 to_frame 下

        points: np.ndarray shape=(n, 3)
        """
        points = points.T  # 转置  (3, n)
        ones = np.ones((1, points.shape[1]))
        points = np.vstack((points, ones))  # (4, n)
        # 转换
        new_points = np.matmul(np.linalg.inv(self.matrix), points)[:-1, :]   # (3, n)
        return new_points.T  # 转置  (n, 3)
   
    @property
    def matrix(self):
        """:obj:`numpy.ndarray` of float: The canonical 4x4 matrix
        representation of this transform.

        The first three columns contain the columns of the rotation matrix
        followed by a zero, and the last column contains the translation vector
        followed by a one.
        """
        return np.r_[np.c_[self._rotation, self._translation], [[0,0,0,1]]]

    @property
    def x_axis(self):
        """:obj:`numpy.ndarray` of float: 不考虑平移，to_frame的x轴在from_frame各坐标轴的投影，即to_frame坐标系中[1,0,0]在from_frame坐标系中的点坐标
        """
        return self.rotation[:,0]

    @property
    def y_axis(self):
        """:obj:`numpy.ndarray` of float: 不考虑平移，to_frame的y轴在from_frame各坐标轴的投影，即to_frame坐标系中[0,1,0]在from_frame坐标系中的点坐标
        """
        return self.rotation[:,1]

    @property
    def z_axis(self):
        """:obj:`numpy.ndarray` of float: 不考虑平移，to_frame的z轴在from_frame各坐标轴的投影，即to_frame坐标系中[0,0,1]在from_frame坐标系中的点坐标
        """
        return self.rotation[:,2]

    def interpolate_with(self, other_tf, t):
        """Interpolate with another rigid transformation.

        Parameters
        ----------
        other_tf : :obj:`RigidTransform`
            The transform to interpolate with.

        t : float
            The interpolation step in [0,1], where 0 favors this RigidTransform.

        Returns
        -------
        :obj:`RigidTransform`
            The interpolated RigidTransform.

        Raises
        ------
        ValueError
            If t isn't in [0,1].
        """
        if t < 0 or t > 1:
            raise ValueError('Must interpolate between 0 and 1')

        interp_translation = (1.0 - t) * self.translation + t * other_tf.translation
        interp_rotation = transformations.quaternion_slerp(self.quaternion, other_tf.quaternion, t)
        interp_tf = RigidTransform(rotation=interp_rotation, translation=interp_translation,
                                  from_frame = self.from_frame, to_frame = self.to_frame)
        return interp_tf

    def linear_trajectory_to(self, target_tf, traj_len):
        """Creates a trajectory of poses linearly interpolated from this tf to a target tf.

        Parameters
        ----------
        target_tf : :obj:`RigidTransform`
            The RigidTransform to interpolate to.
        traj_len : int
            The number of RigidTransforms in the returned trajectory.

        Returns
        -------
        :obj:`list` of :obj:`RigidTransform`
            A list of interpolated transforms from this transform to the target.
        """
        if traj_len < 0:
            raise ValueError('Traj len must at least 0')
        delta_t = 1.0 / (traj_len + 1)
        t = 0.0
        traj = []
        while t < 1.0:
            traj.append(self.interpolate_with(target_tf, t))
            t += delta_t
        traj.append(target_tf)
        return traj

    def dot(self, other_tf):
        """Compose this rigid transform with another.

        将输入的转换与self转换连接(self转换在乘法的左侧)
        self.from -> self.to(other.from) -> other.to

        Parameters
        ----------
        other_tf : :obj:`RigidTransform`
            The other RigidTransform to compose with this one.

        Returns
        -------
        :obj:`RigidTransform`
            A RigidTransform that represents the composition.

        Raises
        ------
        ValueError
            If the to_frame of other_tf is not identical to this transform's
            from_frame.
        """
        if other_tf.from_frame != self.to_frame:
            raise ValueError('from frame of right hand side ({0}) must match to frame of left hand side ({1})'.format(other_tf.from_frame, self.to_frame))

        pose_tf = self.matrix.dot(other_tf.matrix)
        rotation, translation = RigidTransform.rotation_and_translation_from_matrix(pose_tf)

        return RigidTransform(rotation, translation, 
                              from_frame=self.from_frame,
                              to_frame=other_tf.to_frame)

    def inverse(self):
        """Take the inverse of the rigid transform.

        Returns
        -------
        :obj:`RigidTransform`
            The inverse of this RigidTransform.
        """
        inv_pose = np.linalg.inv(self.matrix)
        rotation, translation = RigidTransform.rotation_and_translation_from_matrix(inv_pose)
        return RigidTransform(rotation, translation,
                              from_frame=self._to_frame,
                              to_frame=self._from_frame)

    def save(self, filename):
        """Save the RigidTransform to a file.

        The file format is:
        from_frame
        to_frame
        translation (space separated)
        rotation_row_0 (space separated)
        rotation_row_1 (space separated)
        rotation_row_2 (space separated)

        Parameters
        ----------
        filename : :obj:`str`
            The file to save the transform to.

        Raises
        ------
        ValueError
            If filename's extension isn't .tf.
        """
        f = open(filename, 'w')
        f.write('%s\n' %(self._from_frame))
        f.write('%s\n' %(self._to_frame))
        f.write('%f %f %f\n' %(self._translation[0], self._translation[1], self._translation[2]))
        f.write('%f %f %f\n' %(self._rotation[0, 0], self._rotation[0, 1], self._rotation[0, 2]))
        f.write('%f %f %f\n' %(self._rotation[1, 0], self._rotation[1, 1], self._rotation[1, 2]))
        f.write('%f %f %f\n' %(self._rotation[2, 0], self._rotation[2, 1], self._rotation[2, 2]))
        f.close()

    def as_frames(self, from_frame, to_frame):
        """Return a shallow copy of this rigid transform with just the frames
        changed.

        Parameters
        ----------
        from_frame : :obj:`str`
            The new from_frame.

        to_frame : :obj:`str`
            The new to_frame.

        Returns
        -------
        :obj:`RigidTransform`
            The RigidTransform with new frames.
        """
        return RigidTransform(self.rotation, self.translation, from_frame, to_frame)
    

    def __repr__(self):
        out = 'RigidTransform(rotation={0}, translation={1}, from_frame={2}, to_frame={3})'.format(self.rotation,
                self.translation, self.from_frame, self.to_frame)
        return out

    @staticmethod
    def rotation_from_quaternion(q_wxyz):
        """Convert quaternion array to rotation matrix.

        Parameters
        ----------
        q_wxyz : :obj:`numpy.ndarray` of float
            A quaternion in wxyz order.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix made from the quaternion.
        """
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
        R = transformations.quaternion_matrix(q_xyzw)[:3,:3]
        return R

    @staticmethod
    def quaternion_from_axis_angle(v):
        """Convert axis-angle representation to a quaternion vector.

        Parameters
        ----------
        v : :obj:`numpy.ndarray` of float
            An axis-angle representation.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A quaternion vector from the axis-angle vector.
        """
        theta = np.linalg.norm(v)
        if theta > 0:
            v = v / np.linalg.norm(v)
        ax, ay, az = v    
        qx = ax * np.sin(0.5 * theta)
        qy = ay * np.sin(0.5 * theta)
        qz = az * np.sin(0.5 * theta)        
        qw = np.cos(0.5 * theta)
        q = np.array([qw, qx, qy, qz])
        return q
        
    @staticmethod
    def rotation_from_axis_angle(v):
        """Convert axis-angle representation to rotation matrix.

        Parameters
        ----------
        v : :obj:`numpy.ndarray` of float
            An axis-angle representation.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix made from the axis-angle vector.
        """
        return RigidTransform.rotation_from_quaternion(RigidTransform.quaternion_from_axis_angle(v))
        
    @staticmethod
    def rotation_and_translation_from_matrix(matrix):
        """Helper to convert 4x4 matrix to rotation matrix and translation vector.

        Parameters
        ----------
        matrix : :obj:`numpy.ndarray` of float
            4x4 rigid transformation matrix to be converted.

        Returns
        -------
        :obj:`tuple` of :obj:`numpy.ndarray` of float
            A 3x3 rotation matrix and a 3-entry translation vector.

        Raises
        ------
        ValueError
            If the incoming matrix isn't a 4x4 ndarray.
        """
        if not isinstance(matrix, np.ndarray) or \
                matrix.shape[0] != 4 or matrix.shape[1] != 4:
            raise ValueError('Matrix must be specified as a 4x4 ndarray')
        rotation = matrix[:3,:3]
        translation = matrix[:3,3]
        return rotation, translation

    @staticmethod
    def x_axis_rotation(theta):
        """Generates a 3x3 rotation matrix for a rotation of angle
        theta about the x axis.

        Parameters
        ----------
        theta : float
            amount to rotate, in radians

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        R = np.array([[1, 0, 0,],
                      [0, np.cos(theta), -np.sin(theta)],
                      [0, np.sin(theta), np.cos(theta)]])
        return R

    @staticmethod
    def y_axis_rotation(theta):
        """Generates a 3x3 rotation matrix for a rotation of angle
        theta about the y axis.

        Parameters
        ----------
        theta : float
            amount to rotate, in radians

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        R = np.array([[np.cos(theta), 0, np.sin(theta)],
                      [0, 1, 0],
                      [-np.sin(theta), 0, np.cos(theta)]])
        return R

    @staticmethod
    def z_axis_rotation(theta):
        """Generates a 3x3 rotation matrix for a rotation of angle
        theta about the z axis.

        Parameters
        ----------
        theta : float
            amount to rotate, in radians

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0],
                      [0, 0, 1]])
        return R

    @staticmethod
    def random_rotation():
        """Generates a random 3x3 rotation matrix with SVD.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A random 3x3 rotation matrix.
        """
        rand_seed = np.random.rand(3, 3)
        U, S, V = np.linalg.svd(rand_seed)
        return U

    @staticmethod
    def random_translation():
        """Generates a random translation vector.

        Returns
        -------
        :obj:`numpy.ndarray` of float
            A 3-entry random translation vector.
        """
        return np.random.rand(3)

    @staticmethod
    def load(filename):
        """Load a RigidTransform from a file.

        The file format is:
        from_frame
        to_frame
        translation (space separated)
        rotation_row_0 (space separated)
        rotation_row_1 (space separated)
        rotation_row_2 (space separated)

        Parameters
        ----------
        filename : :obj:`str`
            The file to load the transform from.

        Returns
        -------
        :obj:`RigidTransform`
            The RigidTransform read from the file.

        Raises
        ------
        ValueError
            If filename's extension isn't .tf.
        """
        file_root, file_ext = os.path.splitext(filename)

        f = open(filename, 'r')
        lines = list(f)
        from_frame = lines[0][:-1]
        to_frame = lines[1][:-1]

        t = np.zeros(3)
        t_tokens = lines[2][:-1].split()
        t[0] = float(t_tokens[0])
        t[1] = float(t_tokens[1])
        t[2] = float(t_tokens[2])

        R = np.zeros([3,3])
        r_tokens = lines[3][:-1].split()
        R[0, 0] = float(r_tokens[0])
        R[0, 1] = float(r_tokens[1])
        R[0, 2] = float(r_tokens[2])

        r_tokens = lines[4][:-1].split()
        R[1, 0] = float(r_tokens[0])
        R[1, 1] = float(r_tokens[1])
        R[1, 2] = float(r_tokens[2])

        r_tokens = lines[5][:-1].split()
        R[2, 0] = float(r_tokens[0])
        R[2, 1] = float(r_tokens[1])
        R[2, 2] = float(r_tokens[2])
        f.close()
        return RigidTransform(rotation=R, translation=t,
                              from_frame=from_frame,
                              to_frame=to_frame)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return hash(self) == hash(other)
        return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(str(self.__dict__))
