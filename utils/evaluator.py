import numpy as np
import pandas as pd
from tqdm import tqdm



class BASEevaluator:

    def __init__(self, prediction, trues, task='TUAR'):

        self.prediction = prediction
        self.label = trues
        self.task = task
        self.LABEL_NAMES = self.get_label_names()


    def get_label_names(self):
        if self.task == 'TUAR':
            return ['ART', 'eyem', 'chew', 'shiv', 'musc', 'elec']

        elif self.task == 'FPSM':
            return ['EVT', 'CA', 'DA', 'SS']

        elif self.task == 'ESES':
            return ['SWI']


    def get_metrics(self):

        TP, FP, FN = self.get_all_confusion_matrix()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        precision = {f'BASE_Pre/{label}': value for label, value in zip(self.LABEL_NAMES, precision[0])}
        recall = {f'BASE_Rec/{label}': value for label, value in zip(self.LABEL_NAMES, recall[0])}
        f1_score = {f'BASE_F1/{label}': value for label, value in zip(self.LABEL_NAMES, f1_score[0])}

        return precision, recall, f1_score

    def get_all_confusion_matrix(self):

        num_class = self.prediction.shape[-1]
        TP_all, FP_all, FN_all = np.zeros((1, num_class)), np.zeros((1, num_class)), np.zeros((1, num_class))
        with tqdm(total=len(self.prediction)) as pbar:
            pbar.set_description('BASE metrics evaluating: ')
            for i in range(len(self.prediction)):
                prediction = self.prediction[i]
                label = self.label[i]
                TP_subject, FP_subject, FN_subject = self.compute_confusion_matrix(prediction, label)
                TP_all += TP_subject
                FP_all += FP_subject
                FN_all += FN_subject
                pbar.update(1)

        return TP_all, FP_all, FN_all

    def compute_confusion_matrix(self, prediction, label):

        '''
        prediction: L x C x num_class
        label: L x C x num_class
        return:
            TP: 1 x num_class
            FN: 1 x num_class
            FP: 1 x num_class
        '''

        assert prediction.shape == label.shape, "Error: the shapes of prediction and label are different."

        y_pred = prediction
        TP = (y_pred * label).sum(axis=0).sum(axis=0, keepdims=True)
        FP = y_pred.sum(axis=0).sum(axis=0, keepdims=True) - TP
        FN = label.sum(axis=0).sum(axis=0, keepdims=True) - TP
        return TP, FP, FN

def BASE_metrics_calc(predictions, trues):
    '''
    predictions: L x C x num_class
    trues: L x C x num_class
    return:
        TP: 1 x num_class
        FN: 1 x num_class
        FP: 1 x num_class
    '''
    evaluator = BASEevaluator(predictions, trues)
    precision, recall, f1_score = evaluator.get_metrics()

    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}



class EACSevaluator:

    def __init__(self, prediction, trues, subject_ids, window_ids, task: str = 'TUAR', min_l: int = 25):

        self.prediction = prediction
        self.label = trues
        self.subject_ids = subject_ids
        self.window_ids = window_ids
        self.task = task
        self.LABEL_NAMES = self.get_label_names()
        self.min_l = min_l
        self.num_channel = self.prediction.shape[-2]
        self.num_class = self.prediction.shape[-1]

        self.prediction, self.label = self.reorgnization()

    def get_label_names(self):
        if self.task == 'TUAR':
            return ['ART', 'eyem', 'chew', 'shiv', 'musc', 'elec']

        elif self.task == 'FPSM':
            return ['EVT', 'CA', 'DA', 'SS']

        elif self.task == 'ESES':
            return ['SWI']

    def reorgnization(self):

        subject_list = list(set(self.subject_ids))
        subject_list.sort()

        prediction = []
        label = []
        for i in subject_list:
            indices = np.where(self.subject_ids == i)[0]
            prediction_i = self.prediction[indices][self.window_ids[indices]]
            prediction_i = prediction_i.reshape(-1, *prediction_i.shape[2:])
            trues_i = self.label[indices][self.window_ids[indices]]
            trues_i = trues_i.reshape(-1, *trues_i.shape[2:])

            prediction_i = self.transfer_prediction(prediction_i)
            trues_i = self.transfer_label(trues_i)
            prediction.append(prediction_i)
            label.append(trues_i)

        return prediction, label


    def get_metrics(self):

        TP, FP, FN = self.get_all_confusion_matrix()

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)

        precision = {f'EACS_Pre/{label}': value for label, value in zip(self.LABEL_NAMES, precision)}
        recall = {f'EACS_Rec/{label}': value for label, value in zip(self.LABEL_NAMES, recall)}
        f1_score = {f'EACS_F1/{label}': value for label, value in zip(self.LABEL_NAMES, f1_score)}

        return precision, recall, f1_score

    def get_all_confusion_matrix(self):

        num_class = self.num_class
        TP_all, FP_all, FN_all = np.zeros(num_class), np.zeros(num_class), np.zeros(num_class)
        with tqdm(total=len(self.prediction)) as pbar:
            pbar.set_description('EACS metrics evaluating: ')
            for i in range(len(self.prediction)):
                prediction = self.prediction[i]
                label = self.label[i]
                TP_subject, FP_subject, FN_subject = self.compute_confusion_matrix(prediction, label)
                TP_all += TP_subject
                FP_all += FP_subject
                FN_all += FN_subject
                pbar.update(1)

        return TP_all, FP_all, FN_all

    def compute_confusion_matrix(self, prediction, label):

        # if self.task == 'multi_cls':
        #     label['label'] = label['label'].apply(lambda x: SINGLE_LABEL_TO_MULTI_LABEL[x])
        #     start_class_id = 1
        # if self.task == 'binary_cls':
        #     label['label'] = label['label'].apply(lambda x: SINGLE_LABEL_TO_BINARY_LABEL[x])
        #     start_class_id = 0

        num_class = self.num_class
        TP, FP, FN = np.zeros(num_class), np.zeros(num_class), np.zeros(num_class)
        for i in range(num_class):
            prediction_class_i = prediction[prediction['label'] == i]
            label_class_i = label[label['label'] == i]
            tp, fp, fn = self.compute_single_class_confusion_matrix(prediction_class_i, label_class_i)
            TP[i] += tp
            FP[i] += fp
            FN[i] += fn

        return TP, FP, FN

    def compute_single_class_confusion_matrix(self, prediction, label):

        TP, FP, FN = 0, 0, 0
        for i in range(self.num_channel):
            prediction_channel_i = prediction[prediction['#Channel'] == i]
            label_channel_i = label[label['#Channel'] == i]
            tp, fp, fn = self.compute_single_class_channel_confusion_matrix(prediction_channel_i, label_channel_i)
            TP += tp
            FP += fp
            FN += fn
        return TP, FP, FN

    def compute_single_class_channel_confusion_matrix(self, prediction, label):

        overlap_flag = None
        tp = 0
        fp = 0
        fn = 0

        if label.shape[0] == 0:
            fp +=prediction.shape[0]
            return tp, fp, fn
        else:
            for anchor_s, anchor_e in label.loc[:, ['start', 'end']].values:
                overlap = (prediction['start'] < anchor_e) & (prediction['end'] > anchor_s)
                if overlap_flag is None:
                    overlap_flag = overlap
                else:
                    overlap = (~ overlap_flag) & overlap
                    overlap_flag = overlap_flag | overlap
                prediction_overlap_anchor = prediction[overlap]

                fn_anchor = 1
                for pred_s, pred_e in prediction_overlap_anchor.loc[:, ['start', 'end']].values:
                    overlap_segment = min(pred_e, anchor_e) - max(pred_s, anchor_s)
                    pred_length = pred_e - pred_s
                    anchor_length = anchor_e - anchor_s
                    error_pred_segment = pred_length - overlap_segment
                    # miss_pred_segment = anchor_length - overlap_segment

                    # 在anchor默认没有重叠的pred的情况下，fn默认等于1
                    # 每当anchor与一个pred产生了重叠，则fn减去当前pred所产生的tp
                    fn_anchor = fn_anchor - overlap_segment / anchor_length

                    tp += overlap_segment / anchor_length
                    # FN += miss_pred_segment / anchor_length
                    fp += min(1.0, error_pred_segment / anchor_length)
                    # if error_pred_segment > anchor_length:
                    #     FP += 1.0
                    # else:
                    #     FP += error_pred_segment / anchor_length
                fn += fn_anchor

            non_overlap_FP = (~ overlap_flag).sum()
            fp += non_overlap_FP
            return tp, fp, fn

    def transfer_prediction(self, output):
        # channel_names = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6',
        #                  'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
        #                  'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
        prediction = self.transfer_prediction_multi_cls(output)
        prediction = prediction.sort_values(by=['start', 'end', '#Channel']).reset_index(drop=True)
        prediction = self.remove_short_prediction(prediction)
        return prediction

    def transfer_label(self, label):
        # channel_names = ['FP1-F7', 'F7-T3', 'T3-T5', 'T5-O1', 'FP2-F8', 'F8-T4', 'T4-T6',
        #                  'T6-O2', 'A1-T3', 'T3-C3', 'C3-CZ', 'CZ-C4', 'C4-T4', 'T4-A2',
        #                  'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2']
        event_label = self.transfer_prediction_multi_cls(label)
        event_label = event_label.sort_values(by=['start', 'end', '#Channel']).reset_index(drop=True)
        event_label = self.remove_short_prediction(event_label)
        return event_label

    def remove_short_prediction(self, prediction):
        prediction['diff'] = prediction['end'] - prediction['start']
        prediction = prediction[prediction['diff'] >= self.min_l]
        prediction = prediction.drop('diff', axis=1)
        return prediction.reset_index(drop=True)

    def transfer_prediction_multi_cls(self, output):
        prediction = []
        for i in range(output.shape[1]):
            for j in range(output.shape[2]):
                binary_output = output[:, i, j]
                start_index, end_index = self.binary_to_index_pairs(binary_output)
                for s, e in zip(start_index, end_index):
                    pred_item = [i, s, e, j]
                    prediction.append(pred_item)
        prediction = pd.DataFrame(prediction, columns=['#Channel', 'start', 'end', 'label'])
        return prediction

    def binary_to_index_pairs(self, binary_output):
        pad_binary_output = np.concatenate([[0], binary_output, [0]])
        start_index = np.where(np.diff(pad_binary_output) == 1)[0]
        end_index = np.where(np.diff(pad_binary_output) == -1)[0]
        return start_index, end_index

def EACS_metrics_calc(predictions, trues, subject_ids, window_ids):
    '''
    predictions: N x L x C x num_class
    trues: N x L x C x num_class
    return:
        TP: 1 x num_class
        FN: 1 x num_class
        FP: 1 x num_class
    '''
    evaluator = EACSevaluator(predictions, trues, subject_ids, window_ids)
    precision, recall, f1_score = evaluator.get_metrics()

    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


if __name__ == '__main__':

    # BASE metrics
    predictions = (np.random.rand(100, 2048, 23, 6) > 0.5).astype(int)
    trues = np.random.randint(0, 2, (100, 2048, 23, 6))
    BASE_metrics = BASE_metrics_calc(predictions, trues)

    # EACS metrics
    subject_ids = np.array([0] * 20 + [1] * 30 + [2] * 10 + [3] * 40)
    window_ids = np.concatenate([np.arange(20), np.arange(30), np.arange(10), np.arange(40)])
    EACS_metrics = EACS_metrics_calc(predictions, trues, subject_ids, window_ids)

    pass