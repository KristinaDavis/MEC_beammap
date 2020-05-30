"""
beammap_bar.py
Kristina Davis
5/24/20

Create a bar of light in the focal plane of MEC using the DM of SCExAO.

This module creates the 2D DM maps to be sent to the shared memory buffer of CACAO which interfaces using the
pyMilk library, specifically the SHM module. SHM then interfaces with the  ImageStreamIOWrap sub-module, which
does the bulk of the cython-python formatting of numpy into the correct C-type struct.

Most of the functionality for pymik can be found on the README
https://github.com/milk-org/pyMilk
or else in the SHM code itself
https://github.com/milk-org/pyMilk/blob/master/pyMilk/interfacing/isio_shmlib.py


"""
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyMilk.interfacing.isio_shmlib import SHM


class BarParams():
    def __init__(self, size=[50,50]):
        # Probe Dimensions (extent in pupil plane coordinates)
        # pairwise probes documented in Give'on et al 2011 doi: 10.1117/12.895117
        self.probe_sz = size
        self.dir = 'y'
        self.probe_w = size[0]  # [actuator coordinates] width of the probe (image plane coords?)
        self.probe_h = size[1]  # [actuator coordinates] height of the probe (image plane coords?)
        self.probe_center = [-0,-0]  # [actuator coordinates] center position of the probe
        self.probe_amp = 0.20  # [m] probe amplitude in um, scale should be in units of actuator height limits

        # Probe Motion
        self.phs_intervals = np.pi / 4  # [rad] phase interval over [0, 2pi]
        self.phase_list = np.arange(0, 2 * np.pi, self.phs_intervals)  # FYI not inclusive of 2pi endpoint
        self.n_probes = len(self.phase_list)  # number of phase probes
        self.phase_integration_time = 0.01  # [s]
        self.null_time = 0.1  # [s]
        self.probe_type = "pairwise"


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


class AppliedProbe():
    """
    way to save probes. perhaps unnecessary, or convert to an array of instances of img
    """
    def __init__(self, bp, probe, time):
        self.probe = probe
        self.tstamp = time


def MEC_ISIO():
    """
    structures the 2D DM probe in a CACAO-readable structure.



    :return:
    """
    # Create settings instance
    sp = ShmParams()

    # Create Shared Memory Struct
    MECshm = SHM(sp.shm_name)
    data = MECshm.get_data()

    bp = BarParams(data.shape)

    # Create Probe
    # for loop here?
    probe = beambar(bp, dir=bp.dir, center=bp.probe_center, debug=True)

    # Apply Probe
    MECshm.set_data(probe)
    # Read Time
    t_sent = MECshm.IMAGE.md.lastaccesstime

    # Saving Probe and timestamp together
    ap = AppliedProbe(bp, probe, t_sent)

    return MECshm


def beambar(bp, dir='x', center=[0,0], debug=False):
    """
    create a 2D pupil plane pattern that will produce a focal plane bar, either vertical or horizontal

    The probe applied to the DM to achieve CDI is that originally proposed in Giv'on et al 2011, doi: 10.1117/12.895117;
    and was used with proper in Matthews et al 2018, doi:  10.1117/1.JATIS.3.4.045001.

    All that is to say, we apply the CDI probe using the coordinates of the DM actuators,
    and supply the probe height as an additive height to the DM map, which is passed to the prop_dm function.

    :param center: tuple of center coordinates (in pupil actuators) to create the bar
    :return: 2D map of DM coordinates to create the bar
    """
    # Convert actuator centering to pupil plane coords
    cent = np.array(center)/bp.probe_sz

    x = np.linspace(-1/2+cent[0], 1/2+cent[0], bp.probe_sz[0], dtype=np.float32)
    y = np.linspace(-1/2+cent[1], 1/2+cent[1], bp.probe_sz[1], dtype=np.float32)
    X,Y = np.meshgrid(x, y)

    if dir == 'horz' or dir == 'x':
        probe = bp.probe_amp * np.sinc(bp.probe_w * X)
    elif dir == 'vert' or dir == 'y':
        probe = bp.probe_amp * np.sinc(bp.probe_h * Y)
    else:
        raise ValueError("Direction must be 'vert' or 'horz' or 'x' or 'y'")
    # dprint(f"CDI Probe: Min={np.min(probe)*1e9:.2f} nm, Max={np.max(probe)*1e9:.2f} nm")

    # Testing FF propagation
    if debug is True:
        print(f"probe dtype is {probe.dtype}")
        probe_ft = (1/np.sqrt(2*np.pi)) * np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(probe)))
        # probe_ft = 2*np.pi * np.fft.fftshift(np.fft.fft2((probe)))

        fig, ax = plt.subplots(1,3, figsize=(12, 5))
        fig.subplots_adjust(wspace=0.5)
        ax1, ax2, ax3 = ax.flatten()

        im1 = ax1.imshow(probe, interpolation='none', origin='lower')
        ax1.set_title(f"Probe on DM \n(dm coordinates)")
        cb = fig.colorbar(im1, ax=ax1)

        ax2.imshow(np.sqrt(probe_ft.imag**2 + probe_ft.real**2), interpolation='none', origin='lower')
        ax2.set_title("Focal Plane Amplitude of Probe")

        ax3.imshow(np.arctan2(probe_ft.imag, probe_ft.real), interpolation='none', origin='lower')
        ax3.set_title("Focal Plane Phase of Probe")

        # ax1.imshow(probe_ft.real, interpolation='none', origin='lower')
        # ax1.set_title(f"Real FFT probe, " + r'$\theta$' )  # + f"={bp.phase_list/np.pi:.2f}" + r'$\pi$'
        #             # vlim=(-1e-6, 1e-6),
        #               colormap="YlGnBu_r")
        # ax2.imshow(probe_ft.imag, interpolation='none', origin='lower')
        # ax2.set_title(f"Imag FFT probe, " + r'$\theta$')

        plt.show()

    return probe


def move_bar():
    """
    generate a new probe pattern as a series of steps across the DM surface

    :return: phase_list  array of phases to use in CDI probes
    """
    phase_series = np.zeros(sp.numframes) * np.nan

    # Repeating Probe Phases for Integration time
    if bp.phase_integration_time > sp.sample_time:
        phase_hold = bp.phase_integration_time / sp.sample_time
        phase_1cycle = np.repeat(bp.phase_list, phase_hold)
    elif bp.phase_integration_time == sp.sample_time:
        phase_1cycle = bp.phase_list
    else:
        raise ValueError(f"Cannot have CDI phase probe integration time less than sp.sample_time")

    # Repeating Cycle of Phase Probes for Simulation Duration
    full_simulation_time = sp.numframes * sp.sample_time
    time_for_one_cycle = len(phase_1cycle) * bp.phase_integration_time + bp.null_time
    n_phase_cycles = full_simulation_time / time_for_one_cycle
    print(f"number of phase cycles = {n_phase_cycles}")
    if n_phase_cycles < 0.5:
        if bp.n_probes > sp.numframes:
            warnings.warn(f"Number of timesteps in sp.numframes is less than number of CDI phases \n"
                          f"not all phases will be used")
            phase_series = phase_1cycle[0:sp.numframes]
        else:
            warnings.warn(f"Total length of CDI integration time for all phase probes exceeds full simulation time \n"
                          f"Not all phase probes will be used")
            phase_series = phase_1cycle[0:sp.numframes]
    elif 0.5 < n_phase_cycles < 1:
        phase_series[0:len(phase_1cycle)] = phase_1cycle
        print(f"phase_seris  = {phase_series}")
    else:
        n_full = np.floor(n_phase_cycles)
        raise NotImplementedError(f"Whoa, not implemented yet. Hang in there")
        # TODO implement

    return phase_series



def config_shm(shm_name="MECshm"):
    """
    configures the shared memory buffer to write 2D images to CACAO

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
    # bp = BarParams()
    # beambar(bp, center=(0,0), debug=True)

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