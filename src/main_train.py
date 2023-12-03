from main import train
import math

times = [1]

rew_pre_line = [0.5]

batch_s = [384]

gamma = [0.99]

scor_k = [5]

lrr = [0.003]

rep_buf_s = [250000]


def epsi_func(now_episode):
    return max(0.01, 0.5 * (math.tanh(-0.01 * now_episode + 2.5) + 1))


count = 0
for x1 in rew_pre_line:
    for x2 in batch_s:
        for x3 in gamma:
            for x4 in scor_k:
                for x5 in lrr:
                    for x6 in rep_buf_s:
                        count += 1
                        for tt in times:
                            print(f"<<{count}:{tt}>>")

                            train(
                                episode=1000,
                                epsilon_func=epsi_func,
                                gamma=x3,
                                lr=x5,
                                reward_per_line=x1,
                                replay_buffer_size=x6,
                                batch_size=x2,
                                score_k=x4,
                            )

                            # try:
                            #     train(
                            #         episode=1000,
                            #         epsilon_func=epsi_func,
                            #         gamma=x3,
                            #         lr=x5,
                            #         reward_per_line=x1,
                            #         replay_buffer_size=x6,
                            #         batch_size=x2,
                            #         score_k=x4,
                            #     )
                            # except:
                            #     print("")
                            #     print("Err!")
                            # else:
                            #     print("")
                            #     print("Success!")
                            # print("-------------------------------")
