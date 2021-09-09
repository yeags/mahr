import serial
from time import sleep
import  numpy as np
import re
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

class Mahr(serial.Serial):
    ENCODING = 'latin1'
    def __init__(self, com_port, baudrate, units='metric'):
        self.__parameterstatus__ = False
        self.raw_data = None
        self.scanned_profile = None
        self.primary_profile = None
        self.waviness_profile = None
        self.roughness_profile = None
        self.dx = 0.0005
        self.ls = 0.0025
        self.lc = 0.8
        self.alpha = np.sqrt(np.log(2)/np.pi)
        self.re_integer_match = re.compile(r'-?\d+')
        self.vertical_resolution = 200/65536
        self.units = units
        self.generate_cutoff_kernals()
        try:
            super().__init__(port=com_port, baudrate=baudrate, xonxoff=False, write_timeout=3)
            self.initialize()
            sleep(1)
        except serial.SerialException:
            print(f'Cannot establish serial connection on {com_port}.')

    def initialize(self):
        self.send_break()
        self.write('EXT MODE 23\r'.encode(self.ENCODING))
        if self.units.lower() == 'metric':
            self.write('EXT UNIT 0\r'.encode(self.ENCODING))
        elif self.units.lower() == 'imperial':
            self.write('EXT UNIT 1\r'.encode(self.ENCODING))
        else:
            print('Closing serial connection.')
            self.close()
            raise Exception("Units argument must be either 'metric' or 'imperial'.")
    
    def set_parameters(self, Lt, Lc, n_samples):
        '''
        Acceptable values for
        Lt = -1, 1.75, 5.6, 17.5 (tracing length in mm. -1 is AUTO)
        Lc = -1, 0 (0 = standard cutoff. -1 = short)
        n_samples = 1, 2, 3, 4, 5 (number of sampling lengths)
        '''
        self.lt = Lt
        self.num_samples = n_samples
        self.write(f'LT {Lt}\r'.encode(self.ENCODING))
        self.write(f'LC {Lc}\r'.encode(self.ENCODING))
        self.write(f'Z {n_samples}\r'.encode(self.ENCODING))
        self.__parameterstatus__ = True

    def measure(self):
        self.flush()
        self.write('STAP\r'.encode(self.ENCODING))
        sleep(15)
        header = self.read_all()
        print(header.decode(self.ENCODING))
        self.write('PR 4 0\r'.encode(self.ENCODING))
        sleep(1)
        self.flush()
        self.write('PR 1\r'.encode(self.ENCODING))
        sleep(10)
        raw_data_string = self.read_until('}\r\n'.encode(self.ENCODING)).decode(self.ENCODING)
        int_list_string = re.findall(self.re_integer_match, raw_data_string)
        self.raw_data = np.array([int(i) for i in int_list_string])
        print(self.raw_data)


    def save_measurement(self, filename):
        '''
        Saves profile data (n, 5) in the following format:
        raw scanned primary waviness roughness
        '''
        data = np.array((self.raw_data, self.scanned_profile,
                         self.primary_profile, self.waviness_profile,
                         self.roughness_profile)).T
        np.savetxt(filename, data, header='raw scanned primary waviness roughness')

    def create_profiles(self):
        if self.raw_data is not None:
            pre_sp = self.raw_data * self.vertical_resolution
            self.scanned_profile = pre_sp - pre_sp.mean()
            pre_pp_mb, pre_pp = self.remove_trend(self.scanned_profile)
            self.primary_profile = np.convolve(pre_pp, self.sx_ls, mode='same')
            self.waviness_profile = np.convolve(self.primary_profile, self.sx_lc, mode='same')
            self.roughness_profile = self.primary_profile - self.waviness_profile
            self.create_wavelength_data()
        else:
            try:
                self.measure()
                self.create_profiles()
            except:
                print('An error occurred.')
    
    def generate_cutoff_kernals(self):
        x_lc = np.arange(-self.lc, self.lc+self.dx, self.dx)
        x_ls = np.arange(-self.ls, self.ls+self.dx, self.dx)
        sx_lc = (1/self.alpha*self.lc)*np.exp(-np.pi*(x_lc/(self.alpha*self.lc))**2)
        sx_ls = (1/self.alpha*self.ls)*np.exp(-np.pi*(x_ls/(self.alpha*self.ls))**2)
        self.sx_lc = sx_lc/np.sum(sx_lc)
        self.sx_ls = sx_ls/np.sum(sx_ls)

    def create_wavelength_data(self):
        fs = fftfreq(int(self.lt/self.dx + 1), self.dx)
        mask = fs > 0
        fft_values = fft(self.primary_profile)
        fft_theo = 2*np.abs(fft_values/(self.lt/self.dx))
        self.x_wlc = 1/fs[mask]
        self.y_wlc = fft_theo[mask]

    def plot_results(self, savefile=None):
        fig1, (sp, pp) = plt.subplots(2,1, figsize=(11,8.5), sharex=True, grid=True)
        fig2, (rp, wlc) = plt.subplots(2,1, figsize=(11,8.5), sharex=True, grid=True)
        x_ticks = [i*self.lc for i in range(self.num_samples+3)]
        x = np.arange(self.dx, self.lt + self.dx, self.dx)
        sp.plot(x, self.scanned_profile, label='Scanned Profile')
        pp.plot(x, self.primary_profile, label='Primary Profile')
        pp.plot(x, self.waviness_profile, label='Waviness Profile')
        rp.plot(x, self.roughness_profile, label='Roughness Profile')
        wlc.plot(self.x_wlc, self.y_wlc, label='Wavelength Content')
        wlc.set_xscale('log')
        sp.set_xticks(x_ticks)
        pp.set_xticks(x_ticks)
        rp.set_xticks(x_ticks)
        sp.legend()
        pp.legend()
        rp.legend()
        if savefile is not None:
            fig1.savefig(f'Scanned and Primary Profiles - {savefile}.pdf')
            fig2.savefig(f'Roughness and Wavelength Profiles - {savefile}.pdf')
        plt.show()

    def remove_trend(self, data):
        x = np.arange(self.dx, self.lt + self.dx, self.dx)
        x_m = np.mean(x)
        y_m = np.mean(data)
        x_dev = x - x_m
        y_dev = data - y_m
        xy_dev = x_dev * y_dev
        xsq_dev = x_dev**2
        m = np.sum(xy_dev)/np.sum(xsq_dev)
        b = y_m - m * x_m
        trend_removed = data - m*x - b
        return (np.array([m, b]), trend_removed)
    

if __name__ == '__main__':
    mahr = Mahr('COM3', 19200)
    mahr.set_parameters(5.6, 0, 5)
    mahr.measure()
    mahr.create_profiles()
    mahr.plot_results(savefile='Brass Sample')
    mahr.save_measurement('brass_sample.txt')
    mahr.close()