"""
beammap_bar.py
Kristina Davis
5/24/20

Create a bar of light in the focal plane of MEC using the DM of SCExAO.

This module creates the 2D DM maps to be sent to the shared memory buffer of CACAO which interfaces using the
pyMilk library, specifically the SHM module. SHM then interfaces with the  ImageStreamIOWrap sub-module, which
does the bulk of the cython-python formatting of numpy into the correct C-type struct.

Most of the functionality for pyMilk can be found on the README
https://github.com/milk-org/pyMilk
or else in the SHM code itself
https://github.com/milk-org/pyMilk/blob/master/pyMilk/interfacing/isio_shmlib.py

"""
import numpy as np
import matplotlib.pyplot as plt
from pyMilk.interfacing.isio_shmlib import SHM


class ShmParams():
    """
    settings related to the shared memory file in pyMilk. The SHM has the following format (documentation from
    the pyMilk SHM code itself https://github.com/milk-org/pyMilk/blob/master/pyMilk/interfacing/isio_shmlib.py

    SHM(fname: str,                     fname: name of the shm file
                                        the resulting name will be $MILK_SHM_DIR/<fname>.im.shm
        data: numpy.ndarray = None,     a numpy array (1, 2 or 3D of data)
                                        alternatively, a tuple ((int, ...), dtype) is accepted
                                        which provides shape and datatype (string or np type)
        nbkw: int = 0,                  number of keywords to be appended to the data structure (optional)
        shared: bool = True,            True if the memory is shared among users
        location: int = -1,             -1 for CPU RAM, >= 0 provides the # of the GPU.
        verbose: bool = False,          arguments "packed" and "verbose" are unused.
        packed=False,                   arguments "packed" and "verbose" are unused.
        autoSqueeze: bool = True,       Remove singleton dimensions between C-side and user side. Otherwise, 
                                        we take the necessary steps to squeeze / pad the singleton dimensions.
                                        Warning: [data not None] assumes [autoSqueeze=False].
                                        If you're creating a SHM, it's assumed you know what you're passing.
        symcode: int = 4                A symcode parameter enables to switch through all transposed-flipped 
                                        representations of rectangular images. The default is 4 as to provide 
                                        retro-compatibility. We currently lack compatibility with fits files.
        )
    """
    def __init__(self):
        # self.shm_name = 'MECshm'  # use default cacao shm channel for speckle nulling
        self.shm_name = 'dm00disp06'  # use default cacao shm channel for speckle nulling

        self.shm_buffer = np.empty((50, 50), dtype=np.float32)  # this is where the probe pattern goes, so allocate the appropriate size
        self.location = -1  # -1 for CPU RAM, >= 0 provides the # of the GPU.
        self.shared = True  # if true then a shared memory buffer is allocated. If false, only local storage is used.


class BarParams():
    def __init__(self, size=[50,50]):
        # Probe Dimensions (extent in pupil plane coordinates)
        # pairwise probes documented in Give'on et al 2011 doi: 10.1117/12.895117
        self.probe_sz = size
        self.dir = 'horz'
        self.probe_w = size[0]  # [actuator coordinates] width of the probe (image plane coords?)
        self.probe_h = size[1]  # [actuator coordinates] height of the probe (image plane coords?)
        self.probe_center = [0,0]  # [actuator coordinates] center position of the probe
        self.probe_amp = 0.10  # [m] probe amplitude in um, scale should be in units of actuator height limits
        self.debug = False


class AppliedProbe:
    """
    way to save probes. perhaps unnecessary, or convert to an array of instances of img
    """
    def __init__(self, bp, probe, time):
        self.probe = probe
        self.tstamp = time
        self.bar_params = bp


def MEC_ISIO():
    """
    interfaces with the shared memory buffer of remote DM

    Here we create the interface between the shm and this code to apply new offsets for the remote DM. The interfacing
    is handled by pyMILK found https://github.com/milk-org/pyMilk. We also create the probe by calling beambar, and
    send that offset map to the shm. We read in the time the probe pattern was applied. Functinality exists to save
    the probe pattern and timestamp together, but it is unused for beammapping on MEC as of 6/4/20 so it is currently
    unused.

    :return: nothing explicitly returned but probe is applied (will persist on DM until it is externally cleared,
            eg by the RTC computer on SCExAO). Saving capability not currently implemented.
    """
    # Create shared memory (shm) interface
    sp = ShmParams()
    MECshm = SHM(sp.shm_name)  # create c-type interface using pyMilk's ISIO wrapper
    data = MECshm.get_data()  # used to determine size of struct (removes chance of creating wrong probe size)

    # Create Probe
    bp = BarParams(data.shape)  # import settings
    probe = beambar(bp, line_dir=bp.dir, center=bp.probe_center, debug=bp.debug)  # create probe pattern
    MECshm.set_data(probe)  # Apply Probe
    t_sent = MECshm.IMAGE.md.lastaccesstime      # Read Time

    # Saving Probe and timestamp together
    ap = AppliedProbe(bp, probe, t_sent)

    return MECshm


def beambar(bp, line_dir='x', center=[0,0], debug=False):
    """
    create a 2D pupil plane pattern that will produce a focal plane bar, either vertical or horizontal

    There are 2 modes of the beambar that we will use for beammapping MEC. The first mode is to have a single bar
    in the center of the focal plane (then scan this with the conex mirror). This is achieved by a sinc function
    on the DM, and you can automatically generate this by setting the center coordinate to zero in the direction
    of the bar (either vert/horz or x/y) to 0. If the center coordinate is not zero, then you enter in the
    coordinate that you want to offset the bar's position from center, in actuator coordinate units. For our purposes,
    if you want to have the bar be located at either end of the focal plane, set the center coordinate to be
    n_actuators/2 which is +/-25 for SCExAO.

    The probe applied to the DM to achieve a bar symmetrically split on either side of the DM is that originally
    proposed for pairwise probing (CDI and EFC) in Giv'on et al 2011, doi: 10.1117/12.895117 and was used
    with proper in Matthews et al 2018, doi:  10.1117/1.JATIS.3.4.045001.

    :param bp: the class instance of BarParams, which holds all the user-defined attributes of the bar
    :param dir: direction of the bar, either 'horz' 'vert' 'x' 'y'
    :param center: tuple of center coordinates (in pupil actuators) to create the bar
    :return: 2D map of DM coordinates to create the bar
    """
    # Setting up coordinate system (in units of DM actuators)
    x = np.linspace(-1/2, 1/2, bp.probe_sz[0], dtype=np.float32)
    y = np.linspace(-1/2, 1/2, bp.probe_sz[1], dtype=np.float32)
    X,Y = np.meshgrid(x, y)

    # Selecting probe type based on direction and single or double bar
    if line_dir == 'horz' and center[1] == 0 or line_dir == 'x' and center[1] == 0:
        probe = bp.probe_amp * np.sinc(bp.probe_w * X)
    elif line_dir == 'horz' and center[1] != 0 or line_dir == 'x' and center[1] != 0:
        bp.probe_h = 1
        probe = bp.probe_amp * np.sinc(bp.probe_w * X) * np.sinc(bp.probe_h * Y) \
                * np.sin(2*np.pi*center[1]*Y + bp.theta)
    elif line_dir == 'vert' and center[0] == 0 or line_dir == 'y' and center[0] == 0:
        probe = bp.probe_amp * np.sinc(bp.probe_h * Y)
        # print(f'help theres an error, center[1] = {center[1]}')
    elif line_dir == 'vert' and center[0] != 0 or line_dir == 'y' and center[0] != 0:
        # probe = sig.sawtooth(Y) * np.sin(2*np.pi*cent[1]*Y)
        bp.probe_w = 1
        probe = bp.probe_amp * np.sinc(bp.probe_w * X) * np.sinc(bp.probe_h * Y) \
                * np.sin(2*np.pi*center[0]*X + bp.theta)
    else:
        raise ValueError("Direction must be 'vert' or 'horz' or 'x' or 'y'")

    # Testing FF propagation
    if debug is True:
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))

        fig, ax = plt.subplots(1,3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()

        # fig.suptitle(f"bp.probe_amp * np.sinc(bp.probe_h * Y)")

        im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        cb = fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(np.sqrt(probe_ft.imag**2 + probe_ft.real**2), interpolation='none', origin='lower')
        ax2.set_title("Focal Plane Amplitude")
        cb = fig.colorbar(im2, ax=ax2)

        ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', origin='lower', cmap='hsv')
        ax3.set_title("Focal Plane Phase")

        # ax1.imshow(probe_ft.real, interpolation='none', origin='lower')
        # ax1.set_title(f"Real FFT probe, " + r'$\theta$' )  # + f"={bp.phase_list/np.pi:.2f}" + r'$\pi$'
        #             # vlim=(-1e-6, 1e-6),
        #               colormap="YlGnBu_r")
        # ax2.imshow(probe_ft.imag, interpolation='none', origin='lower')
        # ax2.set_title(f"Imag FFT probe, " + r'$\theta$')

        plt.show()

    return probe


def new_shm():
    """
    creates a new shm
    WARNING: don't actually use this on the scexao_rtc. Use this to test on your own machine. default location
    is in /tmp/

    :return: img-- the image struct
    """
    import ImageStreamIOWrap as ISIO

    sp = ShmParams()

    img = ISIO.Image()
    img.create(sp.shm_name, sp.shm_buffer, sp.location, sp.shared)

    return img


def example_shm(shm_name="MECshm"):
    """
    example from the isio.shm README found on github

    :param shm_name: name of the shared memory buffer (default is MECshm)
    :return: the struct that contains the shared memory buffer
    """
    MECshm = SHM(shm_name)
    data = MECshm.get_data(check=True)  # Wait for semaphore udate, then read

    # Writing to existing shm stream
    MECshm.set_data(np.random.rand(*MECshm.shape).astype(MECshm.nptype))

    # Reading a RT stream
    ocam = SHM('ocam2d')
    output = ocam.multi_recv_data(10000,
                                  outputFormat=1,  # Puts everything in a 3D np.array
                                  monitorCount=True  # Prints a synthesis of counters, ie likely missed frames
                                  )
    print(output.shape)  # 10000 x 120 x 120
    print(output.dtype)  # np.uint16

    # Creating a brand new stream (e.g. 30x30 int16 images)
    shm_wr = SHM('shm_name', data,  # 30x30 int16 np.array
                 location=-1,  # CPU
                 shared=True,  # Shared
                 )
    # Or with shape and type
    shm_wr = SHM('shm_name', ((30, 30), np.int16), ...)

    return MECshm


if __name__ == '__main__':
    print(f"Testing probe")
    MEC_ISIO()

"""
the SHM struct has the following 
Attributes


Methods
get_data(self, check: bool = False, reform: bool = True, sleepT: float = 0.001, timeout: float = 5.,
        copy: bool = True, checkSemAndFlush: bool = True):
        Reads and returns the data part of the SHM file
        Parameters:
        ----------
        - check: boolean (integer supported); if not False, waits image update
        - copy: boolean, if False returns a np.array pointing to the shm, not a copy.
set_data(self, data: np.ndarray, check_dt: bool = False):
        Upload new data to the SHM file.
        Parameters:
        ----------
        - data: the array to upload to SHM
        - check_dt: boolean (default: false) recasts data
multi_recv_data(self, n: int, outputFormat: int = 0, monitorCount: bool = False) 
        returns a Union[List[np.ndarray], np.ndarray]:
        Synchronous read of n successive images in a stream.
        Parameters:
        ----------
        - n: number of frames to read and return
        - outputFormat: flag to indicate what is desired
                        0 for List[np.ndarray]
                        1 for aggregated np.ndarray
        - monitorSem: Monitor and report the counter states when ready to receive - WIP
save_as_fits(self, fitsname: str): exports the data as a fits file
        Parameters:
        ----------
        - fitsname: a filename (overwrite=True)
close: Clean close of a SHM data structure link
read_meta_data(self, verbose: bool = True):   prints meta-data (from shm.Image.md) to console 
get_counter: returns self.IMAGE.md.cnt0
get_expt: Returns exposure time (sec) as a float
get_fps: Returns framerate (Hz) as a float
get_ndr: Returns NDR status as an int
get_crop: Return image crop boundaries as Tuple[int, int, int, int]:

================================
the shm.Image has the following

Attributes
    acqtimearray
    cntarray
    flagarray
    kw
    md:   md -> metadata
    memsize
    semReadPID
    semWritePID
    shape
    used
    writetimearray
    
Methods
    write
    create
    open
    destroy
    close 
    copy
    semflush
    sempost
    getsemwatindex
    semwait
    semtimedwait



"""