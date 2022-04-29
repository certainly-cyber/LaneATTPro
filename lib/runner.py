import pickle
import random
import logging
import torch
import numpy as np
from tqdm import tqdm, trange
import choose
from cv2 import imwrite
import cv2
import time
import change_input
from torch.utils.tensorboard import SummaryWriter



class Runner:
    def __init__(self, cfg, exp, device, resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.logger = logging.getLogger(__name__)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        writer = SummaryWriter('runs/LaneATT_tensorboard')
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.train()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
                if(choose.tensorboard_flag):
                    writer.add_scalar('training loss',
                                      loss.item(),
                                      (epoch - 1) * len(train_loader) + i)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        if(choose.live_flag):
            videoCapture = change_input.cameraget()
            # 等待打开摄像头
            time.sleep(2)
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()

        # 这么写可以不断开始循环
        while(choose.number):

            if(choose.live_flag):
                choose.number = 1
                success, frame = videoCapture.read()
                print(success)
                change_input.save_image(frame, change_input.out_pathlive, change_input.name)

            else:
                choose.number = 0


            if on_val:
                dataloader = self.get_val_dataloader()
            else:
                dataloader = self.get_test_dataloader()

            test_parameters = self.cfg.get_test_parameters()
            predictions = []
            self.exp.eval_start_callback(self.cfg)
            fps_all = 0
            fps_number = 0
            # os.remove(r"C:\Users\39232\Desktop\LaneATT-main\datasets\culane\picture.jpg")
            with torch.no_grad():
                # tqdm是进度条
                for idx, (images, _, _) in enumerate(tqdm(dataloader)):
                    start = time.time()
                    images = images.to(self.device)

                    output = model(images, **test_parameters)
                    end = time.time()
                    prediction = model.decode(output, as_lanes=True)

                    predictions.extend(prediction)
                    fps = 1 / (end - start)
                    fps_all = fps_all + fps
                    fps_number = fps_number + 1



                    if self.view:

                        img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                        img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                        # 调整图像
                        # img = cv2.flip(
                        #     img,
                        #     1  # 1：水平镜像，-1：垂直镜像
                        # )
                        if self.view == 'mistakes' and fp == 0 and fn == 0:
                            continue
                        # pred是窗口名字


                        # print(end-start)
                        # print(fps)
                        # 保留两位小数并输出
                        fps = 3.95
                        # cv2.putText(img, "FPS:" + str(round(fps,2)), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0, 0), 1,cv2.LINE_AA)
                        cv2.imshow('pred', img)
                        if(cv2.waitKey(1)&0xFF == ord('q')):
                            # 利用错误退出
                            # waitkey这个是视频输入延时单位是ms
                            return 0
            fps_av = fps_all/fps_number
            print(fps_av)


            # time.sleep(0.01)  # 睡10ms


        if save_predictions:
            with open('predictions.pkl', 'wb') as handle:
                pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)


    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  # 只有win必须是1，linux是8可以
                                                  # num_workers=8,
                                                  shuffle=False,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
if __name__ == '__main__':
    print("runner ok!")