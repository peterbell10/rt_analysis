import numpy as np
from types import SimpleNamespace

def read_rt_data(filename):
    with open(filename, 'rb') as f:
        return _read_rt_data(f)

def _read_rt_data(f):
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

    ver = (hdr[1], hdr[2])
    nflux = hdr[3]
    nlos = hdr[4]

    _names = ['rates', 'velocities', 'ncols', 'refinements', 'single', 'timescales']
    flags = { n: v for n, v in zip(_names, hdr[5:]) }
    flags['dold'] = ver[1] < 3
    flags['equilibrium_values'] = ver >= (4, 0)

    float_dtype = np.float32().newbyteorder(byte_order)
    hdr = np.fromfile(f, dtype=float_dtype, count=128)

    _names = ['expansion_factor', 'redshift', 'time']
    header = { n: v for n, v in zip(_names, hdr) }

    data = [_RtData(nlos, nflux, flags, f, byte_order) for los in range(nlos)]

    return (data, flags, header)


class _RtData:
    def __init__(self, NLOS, NFLUX, flags, f, byte_order):
        real_dtype = np.float32 if flags['single'] else np.float64
        real_dtype = real_dtype().newbyteorder(byte_order)
        uint64 = np.uint64().newbyteorder(byte_order)
        int64 = np.int64().newbyteorder(byte_order)

        self.num_cells, self.cell, self.nbytes = np.fromfile(f, uint64, 3)
        nflux = self.num_cells
        self.R                     = np.fromfile(f, real_dtype, nflux)
        self.dR                    = np.fromfile(f, real_dtype, nflux)
        self.D                     = np.fromfile(f, real_dtype, nflux)

        if flags['dold']:
            self.dold              = np.fromfile(f, real_dtype, nflux)

        self.entropy               = np.fromfile(f, real_dtype, nflux)
        self.T                     = np.fromfile(f, real_dtype, nflux)
        self.n_H                   = np.fromfile(f, real_dtype, nflux)
        self.f_H1                  = np.fromfile(f, real_dtype, nflux)
        self.n_He                  = np.fromfile(f, real_dtype, nflux)
        self.f_He1                 = np.fromfile(f, real_dtype, nflux)
        self.f_He2                 = np.fromfile(f, real_dtype, nflux)
        self.f_He3                 = np.fromfile(f, real_dtype, nflux)
        self.column_H1             = np.fromfile(f, real_dtype, nflux)
        self.column_He1            = np.fromfile(f, real_dtype, nflux)
        self.column_He2            = np.fromfile(f, real_dtype, nflux)

        if flags['rates']:
            # Parts not incluselfed in RT_DATA_SHORT
            self.G                 = np.fromfile(f, real_dtype, nflux)
            self.G_H1              = np.fromfile(f, real_dtype, nflux)
            self.G_He1             = np.fromfile(f, real_dtype, nflux)
            self.G_He2             = np.fromfile(f, real_dtype, nflux)
            self.I_H1              = np.fromfile(f, real_dtype, nflux)
            self.I_He1             = np.fromfile(f, real_dtype, nflux)
            self.I_He2             = np.fromfile(f, real_dtype, nflux)
            self.L                 = np.fromfile(f, real_dtype, nflux)
            self.L_H1              = np.fromfile(f, real_dtype, nflux)
            self.L_He1             = np.fromfile(f, real_dtype, nflux)
            self.L_He2             = np.fromfile(f, real_dtype, nflux)
            self.L_eH              = np.fromfile(f, real_dtype, nflux)
            self.L_C               = np.fromfile(f, real_dtype, nflux)
            self.E_H1              = np.fromfile(f, real_dtype, nflux)
            self.E_He1             = np.fromfile(f, real_dtype, nflux)
            self.E_He2             = np.fromfile(f, real_dtype, nflux)

        if flags['velocities']:
            self.v_z               = np.fromfile(f, real_dtype, nflux)
            self.v_x               = np.fromfile(f, real_dtype, nflux)

        if flags['equilibrium_values']:
            self.n_HI_Equilibrium  = np.fromfile(f, real_dtype, nflux)
            self.n_HII_Equilibrium = np.fromfile(f, real_dtype, nflux)
            self.Gamma_HI          = np.fromfile(f, real_dtype, nflux)

        if flags['timescales']:
            self.TimeScales        = np.fromfile(f, real_dtype, nflux)

        if flags['ncols']:
            self.ncols             = np.fromfile(f, float_dtype, 12)

        if flags['refinements']:
            self.cell_buffer_index = np.fromfile(f, int64, nflux)

if __name__ == '__main__':
    d, f, h = read_rt_data('Rates_t=150.000')
