import os
import numpy as np
import numpy.linalg as npl
import nibabel as nib
import matplotlib.pyplot as plt


class NIFTI_READER():
    def __init__(self, file_path,affine=None):
        @staticmethod
        def is_nifti(file_path):
            try:
                nib.load(file_path)
                return True
            except Exception:
                return False
        self.is_nifti = is_nifti(file_path)
        self.nifti = nib.load(file_path)
        self.data=self.nifti.get_fdata()

        self.original_shape = self.data.shape
        self.is_4d = False

        if self.data.ndim == 4:
            print(f"input nifti is 4D data,shape:{self.data.shape}, only the first volume will be used.")
            self.data = self.data[:,:,:,0]
            #通常4d仿射矩阵是(4,5)，矩阵对应(3d+time),我们只需要3d部分的，就是左上角的3x3矩阵
            self.affine = self.nifti.affine[:4,:4] if self.nifti.affine.shape ==(4,5) else self.nifti.affine
        else:
            self.affine = self.nifti.affine


        # if self.affine[0,0]>0:
        #     self.data = np.flip(self.data,0)
        #     print("flipped data ,id = 0")
        # if self.affine[1,1]>0:
        #     self.data = np.flip(self.data,1)
        #     print("flipped data ,id = 1")
        # if self.affine[2,2]>0:
        #     self.data = np.flip(self.data,2)
        #     print("flipped data ,id = 2")
        codes_labels = self._aff2axcodes(self.affine)
        self.codes = self._axcodes2ornt(codes_labels)
        self.order = np.argsort([c[0] for c in self.codes])

        self.flips = np.array([c[1] < 0 for c in self.codes])[self.order]

    def _axcodes2ornt(self,axcodes, labels=None):
        """Convert axis codes `axcodes` to an orientation

        Parameters
        ----------
        axcodes : (N,) tuple
            axis codes - see ornt2axcodes docstring
        labels : optional, None or sequence of (2,) sequences
            (2,) sequences are labels for (beginning, end) of output axis.  That
            is, if the first element in `axcodes` is ``front``, and the second
            (2,) sequence in `labels` is ('back', 'front') then the first
            row of `ornt` will be ``[1, 1]``. If None, equivalent to
            ``(('L','R'),('P','A'),('I','S'))`` - that is - RAS axes.

        Returns
        -------
        ornt : (N,2) array-like
            orientation array - see io_orientation docstring

        Examples
        --------
        # >>> axcodes2ornt(('F', 'L', 'U'), (('L','R'),('B','F'),('D','U')))
        array([[ 1.,  1.],
               [ 0., -1.],
               [ 2.,  1.]])
        """
        labels = list(zip('LPI', 'RAS')) if labels is None else labels
        allowed_labels = sum(map(list, labels), [None])
        if len(allowed_labels) != len(set(allowed_labels)):
            raise ValueError(f'Duplicate labels in {allowed_labels}')
        if not set(axcodes).issubset(allowed_labels):
            raise ValueError(f'Not all axis codes {list(axcodes)} in label set {allowed_labels}')
        n_axes = len(axcodes)
        ornt = np.ones((n_axes, 2), dtype=np.int8) * np.nan
        for code_idx, code in enumerate(axcodes):
            for label_idx, codes in enumerate(labels):
                if code is None:
                    continue
                if code in codes:
                    if code == codes[0]:
                        ornt[code_idx, :] = [label_idx, -1]
                    else:
                        ornt[code_idx, :] = [label_idx, 1]
                    break
        return ornt
    def _io_orientation(self,affine, tol=None):
        """Orientation of input axes in terms of output axes for `affine`

        Valid for an affine transformation from ``p`` dimensions to ``q``
        dimensions (``affine.shape == (q + 1, p + 1)``).

        The calculated orientations can be used to transform associated
        arrays to best match the output orientations. If ``p`` > ``q``, then
        some of the output axes should be considered dropped in this
        orientation.

        Parameters
        ----------
        affine : (q+1, p+1) ndarray-like
           Transformation affine from ``p`` inputs to ``q`` outputs.  Usually this
           will be a shape (4,4) matrix, transforming 3 inputs to 3 outputs, but
           the code also handles the more general case
        tol : {None, float}, optional
           threshold below which SVD values of the affine are considered zero. If
           `tol` is None, and ``S`` is an array with singular values for `affine`,
           and ``eps`` is the epsilon value for datatype of ``S``, then `tol` set
           to ``S.max() * max((q, p)) * eps``

        Returns
        -------
        orientations : (p, 2) ndarray
           one row per input axis, where the first value in each row is the closest
           corresponding output axis. The second value in each row is 1 if the
           input axis is in the same direction as the corresponding output axis and
           -1 if it is in the opposite direction.  If a row is [np.nan, np.nan],
           which can happen when p > q, then this row should be considered dropped.
        """
        affine = np.asarray(affine)
        q, p = affine.shape[0] - 1, affine.shape[1] - 1
        # extract the underlying rotation, zoom, shear matrix
        RZS = affine[:q, :p]
        zooms = np.sqrt(np.sum(RZS * RZS, axis=0))
        # Zooms can be zero, in which case all elements in the column are zero, and
        # we can leave them as they are
        zooms[zooms == 0] = 1
        RS = RZS / zooms
        # Transform below is polar decomposition, returning the closest
        # shearless matrix R to RS
        P, S, Qs = npl.svd(RS, full_matrices=False)
        # Threshold the singular values to determine the rank.
        if tol is None:
            tol = S.max() * max(RS.shape) * np.finfo(S.dtype).eps
        keep = S > tol
        R = np.dot(P[:, keep], Qs[keep])
        # the matrix R is such that np.dot(R,R.T) is projection onto the
        # columns of P[:,keep] and np.dot(R.T,R) is projection onto the rows
        # of Qs[keep].  R (== np.dot(R, np.eye(p))) gives rotation of the
        # unit input vectors to output coordinates.  Therefore, the row
        # index of abs max R[:,N], is the output axis changing most as input
        # axis N changes.  In case there are ties, we choose the axes
        # iteratively, removing used axes from consideration as we go
        ornt = np.ones((p, 2), dtype=np.int8) * np.nan
        for in_ax in range(p):
            col = R[:, in_ax]
            if not np.allclose(col, 0):
                out_ax = np.argmax(np.abs(col))
                ornt[in_ax, 0] = out_ax
                assert col[out_ax] != 0
                if col[out_ax] < 0:
                    ornt[in_ax, 1] = -1
                else:
                    ornt[in_ax, 1] = 1
                # remove the identified axis from further consideration, by
                # zeroing out the corresponding row in R
                R[out_ax, :] = 0
        return ornt
    def _ornt2axcodes(self,ornt, labels=None):
        """Convert orientation `ornt` to labels for axis directions

        Parameters
        ----------
        ornt : (N,2) array-like
            orientation array - see io_orientation docstring
        labels : optional, None or sequence of (2,) sequences
            (2,) sequences are labels for (beginning, end) of output axis.  That
            is, if the first row in `ornt` is ``[1, 1]``, and the second (2,)
            sequence in `labels` is ('back', 'front') then the first returned axis
            code will be ``'front'``.  If the first row in `ornt` had been
            ``[1, -1]`` then the first returned value would have been ``'back'``.
            If None, equivalent to ``(('L','R'),('P','A'),('I','S'))`` - that is -
            RAS axes.

        Returns
        -------
        axcodes : (N,) tuple
            labels for positive end of voxel axes.  Dropped axes get a label of
            None.

        Examples
        --------
        # >>> ornt2axcodes([[1, 1],[0,-1],[2,1]], (('L','R'),('B','F'),('D','U')))
        ('F', 'L', 'U')
        """
        if labels is None:
            labels = list(zip('LPI', 'RAS'))
        axcodes = []
        for axno, direction in np.asarray(ornt):
            if np.isnan(axno):
                axcodes.append(None)
                continue
            axint = int(np.round(axno))
            if axint != axno:
                raise ValueError(f'Non integer axis number {axno:f}')
            elif direction == 1:
                axcode = labels[axint][1]
            elif direction == -1:
                axcode = labels[axint][0]
            else:
                raise ValueError('Direction should be -1 or 1')
            axcodes.append(axcode)
        return tuple(axcodes)
    def _aff2axcodes(self,aff, labels=None, tol=None):
        """axis direction codes for affine `aff`

        Parameters
        ----------
        aff : (N,M) array-like
            affine transformation matrix
        labels : optional, None or sequence of (2,) sequences
            Labels for negative and positive ends of output axes of `aff`.  See
            docstring for ``ornt2axcodes`` for more detail
        tol : None or float
            Tolerance for SVD of affine - see ``io_orientation`` for more detail.

        Returns
        -------
        axcodes : (N,) tuple
            labels for positive end of voxel axes.  Dropped axes get a label of
            None.

        Examples
        --------
        # >>> aff = [[0,1,0,10],[-1,0,0,20],[0,0,1,30],[0,0,0,1]]
        # >>> aff2axcodes(aff, (('L','R'),('B','F'),('D','U')))
        ('B', 'R', 'U')
        """
        ornt = self._io_orientation(aff, tol)
        return self._ornt2axcodes(ornt, labels)

    def get_slice_array(self,orient,idx):
        """
        :param data: the input numpy array.
        :param orient: the target orientation.
        def by:
            0 --> sag.
            1 --> cor.
            2 --> axi.
        :param idx: the index of the slice.
        :return: the sliced numpy array,the transform info (transpose,flip_x,flip_y)
        """

        slice_2d = np.rollaxis(self.data, axis=self.order[orient])[idx]
        xax = [1, 0, 0][orient]
        yax = [2, 2, 1][orient]
        # remaining_axes = [0, 1, 2]
        # del remaining_axes[orient]
        # xax, yax = remaining_axes
        flip_x = self.flips[xax]
        flip_y = self.flips[yax]
        transpose = self.order[xax] < self.order[yax]


        if transpose:
            slice_2d = slice_2d.T
        if flip_x:
            slice_2d = slice_2d[:, ::-1]
        if flip_y :
            slice_2d = slice_2d[::-1]
        return slice_2d,(transpose,flip_x,flip_y)

    def get_slices_num(self, orient):
        return self.data.shape[self.order[orient]]

    def get_zeros_volume(self):
        return np.zeros_like(self.data)

    def get_slices_shape(self, orient):
        temp,_ = self.get_slice_array(orient,0)
        shape = np.shape(temp)
        del temp
        return shape

    def show_slice(self,orient,idx,cmap='gray'):
        slice_2d,info = self.get_slice_array(orient, idx)
        plt.imshow(slice_2d, cmap=cmap, origin='lower')
        plt.title(f'orient={orient}, idx=60 | transform={info}')
        plt.show()

    def align_to_me(self,orient,mask,idx,transpose_list=None,mask_array=None):
        if transpose_list is None:
            transpose_list = self.get_slice_array(orient,0)[1]
        if mask_array is None:
            mask_array = self.get_zeros_volume()
        transpose = transpose_list[0]
        flip_x = transpose_list[1]
        flip_y = transpose_list[2]
        if flip_y:
            mask = mask[::-1]
        if flip_x:
            mask = mask[:, ::-1]
        if transpose:
            mask = mask.T

        axes = [slice(None)] * 3
        axes[self.order[orient]] = idx
        mask_array[tuple(axes)] = mask

        return mask_array

    def save_seg(self,seg_array,save_path,dtype=np.uint8):
        seg_img = nib.Nifti1Image(seg_array, affine=self.affine)
        nib.save(seg_img, save_path)
        print(f"seg saved to {save_path}")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # brain=NIFTI_READER(r"/path/to/your/example_0000.nii.gz")
    # print(f"tra num:{brain.get_slices_num(0)}")
    # print(f"tra shape:{brain.get_slices_shape(0)}")
    # cor_slice = brain.get_slice_array(0,115)
    # plt.imshow(cor_slice, cmap='gray',origin='lower')
    # plt.show()
    path = r"/path/to/your/example_seg.nii.gz"
    brain_old = nib.load(path)
    print(f"brain_old shape:{brain_old.get_fdata().shape}")

    brain_new = NIFTI_READER(path)
    slice = brain_new.get_slice_array(1,15)[0]
    print(f"slice shape",slice.shape)
    zero = brain_new.get_zeros_volume()
    print(f"brain_new shape:{zero.shape}")

    #%% test 2

    path = r"/path/to/your/example2_0000.nii.gz"
    brain_old = nib.load(path)
    print(f"brain_old shape:{brain_old.get_fdata().shape}")

    brain_new = NIFTI_READER(path)
    slice = brain_new.get_slice_array(1, 15)[0]
    print(f"slice shape", slice.shape)
    zero = brain_new.get_zeros_volume()
    print(f"brain_new shape:{zero.shape}")



