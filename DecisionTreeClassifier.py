from sklearn import tree
from sklearn.metrics import f1_score
import numpy as np
import os

import DataManager as dm


(x_train, y_train), (x_test, y_test) = dm.load_data()

model_params = {'min_samples_leaf': 1, 'min_samples_split': 2, 'max_depth': 7, 'random_state': 3}
print('Model properties: %s' % str(model_params))

model = tree.DecisionTreeClassifier(**model_params)
model.fit(x_train, y_train)
print('Fitting done')
print('Model train score: %.3f' % model.score(x_train, y_train))
print('Model test score: %.3f' % model.score(x_test, y_test))

y_test_predicted = model.predict(x_test)

f_score = f1_score(y_test, y_pred=y_test_predicted, average=None)
print('Model test F1 score:\n\t- normal letter: %.3f\n\t- spam letter: %.3f' % (f_score[0], f_score[1]))

feature_names = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
                 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet',
                 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will',
                 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free',
                 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit',
                 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money',
                 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650',
                 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet' ,'word_freq_857',
                 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology',
                 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct',
                 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project',
                 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference',
                 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$',
                 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest',
                 'capital_run_length_total']
class_names = ['normal', 'spam']


dotfile_path = os.path.join('render', 'tree.dot')
dotfile = open(dotfile_path, 'w')

tree.export_graphviz(model, out_file=dotfile,
                     feature_names=feature_names,
                     class_names=class_names,
                     filled=True, rounded=True,
                     special_characters=True)
dotfile.close()


import pydot as dot
(graph,) = dot.graph_from_dot_file(dotfile_path)

png_file_path = os.path.join('render', 'tree.png')
graph.write_png(png_file_path)
print('Tree saved at "%s"' % png_file_path)
