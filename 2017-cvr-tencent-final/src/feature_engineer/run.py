import subprocess

# cmd = 'python3 分布概率.py'
# subprocess.call(cmd, shell=True)
#
# cmd = 'python3 转化率统计.py'
# subprocess.call(cmd, shell=True)

cmd = 'python3 com_feats.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 clk_cnv_time_interval.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 data_sign.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 user_time_interval_feat.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 user_action_app_sum.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 user_action_installedapps.py'
subprocess.call(cmd, shell=True)

cmd = 'python3 get_click_history_feature_df.py'
subprocess.call(cmd, shell=True)
