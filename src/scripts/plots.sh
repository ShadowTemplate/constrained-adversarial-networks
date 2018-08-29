cd ..
# 2
python plot.py -e bgan_S can_S -l BGANs CANs -t "D: poly20, C: area_convex" -o D_poly20_C_area_convex_N_bgan_can

# 3.a
python plot.py -e bgan_pc_S can_pc_S -l BGANs CANs -t "D: poly20_pc, C: some_pc" -o D_poly20_pc_C_some_pc_N_bgan_can

# 3.b
python plot.py -e can_pc_S can_pc_ll_S -l CANs CANs_out -t "D: poly20_pc, C: some_pc" -o D_poly20_pc_C_some_pc_N_can_canout

# 4
python plot.py -e bgan_pc_S can_pc_S bgan_pc_long_S can_pc_long_S -l BGANs CANs BGANs_long CANs_long -t "D: poly20_pc, C: some_pc" -o D_poly20_pc_C_some_pc_N_bgan_can_bganlong_canlong

# 5
python plot.py -e bgan_pc_all_S can_pc_all_S -l BGANs CANs -t "D: poly20_pc, C: all_pc" -o D_poly20_pc_C_all_pc_N_bgan_can

# 6
python plot.py -e bgan_pc_all_S can_pc_all_S can_pc_all_ll_S can_pc_all_delay_S -l BGANs CANs CANs_out CANs_e100 -t "D: poly20_pc, C: all_pc" -o D_poly20_pc_C_all_pc_N_bgan_can_canout_candelay

#7
python plot.py -e bgan_pc_all_small_S can_pc_all_small_S -l BGANs CANs -t "D: poly20_pc_small, C: all_pc" -o D_poly20_pc_small_C_all_pc_N_bgan_can

#8
# python plot.py -e bgan_pc_all_small_long_S can_pc_all_small_long_S -l BGANs_long CANs_long -t "D: poly20_pc_small, C: all_pc" -o D_poly20_pc_small_C_all_pc_N_bganlong_canlong

