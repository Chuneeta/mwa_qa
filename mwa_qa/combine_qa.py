
import mwa_qa.json_utils as ju
import pandas as pd
import numpy as np

keys = ['OBSID', 'STATUS', 'UNUSED_BLS', 'UNUSED_CHS', 'UNUSED_ANTS',
        'NON_CONVERGED_CHS', 'CONVEREGENCE_VAR'
        'SKEWNESS_UVCUT', 'RMS_AMPVAR_ANT', 'RMS_AMP_FREQ',
        'DFFT_POWER', 'DFFT_POWER_HIGH_PKPL',
        'DFFT_POWER_NKPL', 'RECERIVER_CHISQ', 'MEAN_ALL', 'MEAN_BOX',
        'RMS_ALL', 'RMS_BOX', 'INT_FLUX', 'PEAK_FLUX']


class DataFrameQA(object):
    def __init__(self, cal_jsons=[], img_jsons=[]):
        if type(cal_jsons) == list:
            self.cal_jsons = cal_jsons
        else:
            self.cal_jsons = [cal_jsons]
        if type(img_jsons) == list:
            self.img_jsons = img_jsons
        else:
            self.img_jsons = [img_jsons]
        # assert len(self.cal_jsons) == len(
        #   self.img_jsons), "Length of cal json files should be equal to\
        #       length of image json files"
        self.cal_nfiles = len(self.cal_jsons)
        self.img_nfiles = len(self.img_jsons)

    def read_caljson(self, json, pol):
        data = ju.load_json(json)
        return data['OBSID'], \
            data['STATUS'],\
            data['UNUSED_BLS'], \
            data['UNUSED_CHS'],\
            data['UNUSED_ANTS'], \
            data['NON_CONVERGED_CHS'], \
            data['CONVERGENCE_VAR'], \
            data[pol]['SKEWNESS_UVCUT'], \
            data[pol]['RMS_AMPVAR_ANT'], \
            data[pol]['RMS_AMPVAR_FREQ'], \
            data[pol]['DFFT_POWER'],\
            data[pol]['DFFT_POWER_HIGH_PKPL'],\
            data[pol]['DFFT_POWER_HIGH_NKPL'],\
            data[pol]['RECEIVER_CHISQVAR']

    def read_imgjson(self, json, pol):
        data = ju.load_json(json)
        return str(data[pol]['IMAGE_ID']), \
            data[pol]['MEAN_ALL'],\
            data[pol]['MEAN_BOX'],\
            data[pol]['RMS_ALL'],\
            data[pol]['RMS_BOX'],\
            data[pol]['PKS0023_026']['INT_FLUX'],\
            data[pol]['PKS0023_026']['PEAK_FLUX'],\
            data['V']['MEAN_ALL'],\
            data['V']['MEAN_BOX'],\
            data['V']['RMS_ALL'],\
            data['V']['RMS_BOX'],\
            data['V']['PKS0023_026']['INT_FLUX'],\
            data['V']['PKS0023_026']['PEAK_FLUX']

    def combine_jsons(self, caljson, pol):
        cal_data = self.read_caljson(caljson, pol)
        img_obsids = np.array(self.img_obsids())
        try:
            ind = np.where(img_obsids == cal_data[0])[0]
            ind = ind[0] if len(ind) == 1 else ind
            img_data = self.read_imgjson(self.img_jsons[ind], pol)
            return cal_data + img_data[1::]
        except TypeError:
            return None

    def img_obsids(self):
        obsids = []
        for i in range(self.img_nfiles):
            obsids.append(self.read_imgjson(self.img_jsons[i], 'XX')[0][0:10])
        return obsids

    def cal_obsids(self):
        obsids = []
        for i in range(self.cal_nfiles):
            obsids.append(self.read_caljson(self.cal_jsons[i], 'XX')[0])
        return obsids

    def create_dataframe(self, pol):
        df = pd.DataFrame()
        for i in range(self.cal_nfiles):
            data = self.combine_jsons(self.cal_jsons[i], pol)
            if data is not None:
                df_dict = {}
                for j, key in enumerate(keys):
                    df_dict[key] = data[j]
                df = df.append(df_dict, ignore_index=True)
        return df
