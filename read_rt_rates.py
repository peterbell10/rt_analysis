import numpy as np

def read_rt_rates(filename):
    """Load the RT rates produced by the RT code """
    with open(filename, 'rb') as f:
        return _read_rt_rates(f)

def _read_rt_rates(f):
    bom = f.read(4)
    f.seek(0)

    if np.frombuffer(bom, dtype='<i4', count=1)[0] == 1:
        byte_order = '<'
    elif np.frombuffer(bom, dtype='>i4', count=1)[0] == 1:
        byte_order = '>'
    else:
        raise ValueError(f"Unable to determine byte order in file {filename}")

    # Read the 1024-byte header, checking the byte-order
    # The first half is Integer, the second half floating point
    int_dtype = np.int32().newbyteorder(byte_order)
    hdr = np.fromfile(f, dtype=int_dtype, count=128)

    num_cells = hdr[1];
    flags = {
        'single_precision': hdr[2],
        'cooling': hdr[3]
    }

    float_dtype = np.float32().newbyteorder(byte_order)
    hdr = np.fromfile(f, float_dtype, 128)

    rates = _RtRates(num_cells, flags, f, byte_order)
    return (rates, flags)


class _RtRates:
    """RT rates structure."""
    def __init__(self, num_cells, flags, f, byte_order):
        float_dtype = np.float32 if flags['single_precision'] else np.float64
        float_dtype = float_dtype().newbyteorder(byte_order)

        self.num_cells = num_cells

        self.I_H1	            = np.fromfile(f, float_dtype, num_cells)
        self.I_He1	            = np.fromfile(f, float_dtype, num_cells)
        self.I_He2	            = np.fromfile(f, float_dtype, num_cells)
        self.G	                = np.fromfile(f, float_dtype, num_cells)
        self.G_H1	            = np.fromfile(f, float_dtype, num_cells)
        self.G_He1	            = np.fromfile(f, float_dtype, num_cells)
        self.G_He2	            = np.fromfile(f, float_dtype, num_cells)
        self.alpha_H1           = np.fromfile(f, float_dtype, num_cells)
        self.alpha_He1          = np.fromfile(f, float_dtype, num_cells)
        self.alpha_He2          = np.fromfile(f, float_dtype, num_cells)
        self.L	                = np.fromfile(f, float_dtype, num_cells)
        self.L_H1	            = np.fromfile(f, float_dtype, num_cells)
        self.L_He1	            = np.fromfile(f, float_dtype, num_cells)
        self.L_He2	            = np.fromfile(f, float_dtype, num_cells)

        if flags['cooling']:
            self.L_eH	        = np.fromfile(f, float_dtype, num_cells)
            self.L_C	        = np.fromfile(f, float_dtype, num_cells)

        self.Gamma_HI          = np.fromfile(f, float_dtype, num_cells)
        self.n_HI_Equilibrium  = np.fromfile(f, float_dtype, num_cells)
        self.n_HII_Equilibrium = np.fromfile(f, float_dtype, num_cells)
        self.time_scales       = np.fromfile(f, float_dtype, num_cells)
