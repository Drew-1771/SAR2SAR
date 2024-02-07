from glob import glob

from .utils import *
from .u_net import *


class Denoiser(object):
    def __init__(self, sess, input_c_dim=1, debug=True):
        self.sess = sess
        self.input_c_dim = input_c_dim
        self.Y_ = tf.compat.v1.placeholder(
            tf.float32, [None, None, None, self.input_c_dim], name="clean_image"
        )
        self.Y = autoencoder((self.Y_))
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

        self.debug = debug
        if self.debug:
            print("[*] Initialize model successfully...")

    def load(self, checkpoint_dir):
        if self.debug:
            print(f"[*] Reading checkpoint from {checkpoint_dir}...")
        saver = tf.compat.v1.train.Saver()
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            full_path = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(self.sess, full_path)
            return True
        else:
            return False

    # original function from https://gitlab.telecom-paris.fr/ring/sar2sar
    '''
    def test(self, test_files, ckpt_dir, save_dir, dataset_dir, stride):
        """Test SAR2SAR"""
        tf.initialize_all_variables().run()
        assert len(test_files) != 0, 'No testing data!'
        load_model_status = self.load(ckpt_dir)
        assert load_model_status == True, '[!] Load weights FAILED...'
        print(" [*] Load weights SUCCESS...")
        print("[*] start testing...")
        for idx in range(len(test_files)):
            real_image = load_sar_images(test_files[idx]).astype(np.float32) / 255.0
            # scan on image dimensions
            stride = 64
            pat_size = 256
            # Pad the image
            im_h = np.size(real_image, 1)
            im_w = np.size(real_image, 2)

            count_image = np.zeros(real_image.shape)
            output_clean_image = np.zeros(real_image.shape)

            if im_h==pat_size:
                x_range = list(np.array([0]))
            else:
                x_range = list(range(0, im_h - pat_size, stride))
                if (x_range[-1] + pat_size) < im_h: x_range.extend(range(im_h - pat_size, im_h - pat_size + 1))

            if im_w==pat_size:
                y_range = list(np.array([0]))
            else:
                y_range = list(range(0, im_w - pat_size, stride))
                if (y_range[-1] + pat_size) < im_w: y_range.extend(range(im_w - pat_size, im_w - pat_size + 1))

            for x in x_range:
                for y in y_range:
                    tmp_clean_image = self.sess.run([self.Y], feed_dict={self.Y_: real_image[:, x:x + pat_size,
                                                                                     y:y + pat_size, :]})
                    output_clean_image[:, x:x + pat_size, y:y + pat_size, :] = output_clean_image[:, x:x + pat_size,
                                                                               y:y + pat_size, :] + tmp_clean_image
                    count_image[:, x:x + pat_size, y:y + pat_size, :] = count_image[:, x:x + pat_size, y:y + pat_size,
                                                                        :] + np.ones((1, pat_size, pat_size, 1))
            output_clean_image = output_clean_image/count_image


            noisyimage = denormalize_sar(real_image)
            outputimage = denormalize_sar(output_clean_image)

            imagename = test_files[idx].replace(dataset_dir+"/", "")
            print("Denoised image %s" % imagename)

            save_sar_images(outputimage, noisyimage, imagename, save_dir)
    '''

    def run(
        self,
        input_files,
        save_dir,
        ckpt_dir,
        stride,
        store_noisy: bool,
        generate_png: bool,
    ):
        tf.compat.v1.initialize_all_variables().run()
        assert len(input_files) != 0, "No testing data!"
        load_model_status = self.load(ckpt_dir)
        assert load_model_status == True, "[!] Load weights FAILED..."
        if self.debug:
            print("[*] Load weights SUCCESS...")
            print("[*] start testing...")

        for idx in range(len(input_files)):
            real_image = load_sar_images(input_files[idx]).astype(np.float32) / 255.0
            # scan on image dimensions
            pat_size = 256
            # Pad the image
            im_h = np.size(real_image, 1)
            im_w = np.size(real_image, 2)

            count_image = np.zeros(real_image.shape)
            output_clean_image = np.zeros(real_image.shape)

            if im_h == pat_size:
                x_range = list(np.array([0]))
            else:
                x_range = list(range(0, im_h - pat_size, stride))
                if (x_range[-1] + pat_size) < im_h:
                    x_range.extend(range(im_h - pat_size, im_h - pat_size + 1))

            if im_w == pat_size:
                y_range = list(np.array([0]))
            else:
                y_range = list(range(0, im_w - pat_size, stride))
                if (y_range[-1] + pat_size) < im_w:
                    y_range.extend(range(im_w - pat_size, im_w - pat_size + 1))

            for x in x_range:
                for y in y_range:
                    tmp_clean_image = self.sess.run(
                        [self.Y],
                        feed_dict={
                            self.Y_: real_image[
                                :, x : x + pat_size, y : y + pat_size, :
                            ]
                        },
                    )
                    output_clean_image[:, x : x + pat_size, y : y + pat_size, :] = (
                        output_clean_image[:, x : x + pat_size, y : y + pat_size, :]
                        + tmp_clean_image
                    )
                    count_image[:, x : x + pat_size, y : y + pat_size, :] = count_image[
                        :, x : x + pat_size, y : y + pat_size, :
                    ] + np.ones((1, pat_size, pat_size, 1))
            output_clean_image = output_clean_image / count_image

            noisyimage = denormalize_sar(real_image)
            outputimage = denormalize_sar(output_clean_image)
            if self.debug:
                print("[*] Denoised image %s" % input_files[idx])
            save_sar_images(
                outputimage,
                noisyimage,
                str(Path(input_files[idx]).name),
                save_dir,
                store_noisy,
                generate_png,
                self.debug,
            )


def run_model(
    input_dir: str,
    save_dir: str,
    checkpoint_dir: str = None,
    stride=64,
    store_noisy=True,
    generate_png=True,
    debug=True,
) -> None:
    """
    Runs the despeckling algorithm

    Arguments:
        input_dir: Path to a directory containing the files to be despeckled. Files need to be in .npy
                   format
        save_dir: Path to a directory where the files will be saved
        checkpoint_dir: Path to a directory containing the tensorflow checkpoints, if left as None, the
                        despeckling algorithm will use the grd_checkpoint directory
        stride: U-Net is scanned over the image with a default stride of 64 pixels when the image dimension
                exceeds 256. This parameter modifies the default stride in pixels. Lower pixel count = higher quality
                results, at the cost of higher runtime
        store_noisy: Whether to store the "noisy" or input in the save_dir. Default is True
        generate_png: Whether to generate PNG of the outputs in the save_dir. Default is True
        debug: Whether to generate print statements at runtime that communicate what is going on

    Returns:
        None
    """
    if checkpoint_dir == None:
        checkpoint_dir = Path(__file__).parent / "checkpoint" / "grd_checkpoint"

    input_files = glob(str(Path(input_dir) / "*.npy").format("float32"))
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
    with tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    ) as sess:
        model = Denoiser(sess, debug=debug)
        model.run(
            input_files=input_files,
            save_dir=save_dir,
            ckpt_dir=checkpoint_dir,
            stride=stride,
            store_noisy=store_noisy,
            generate_png=generate_png,
        )
        if debug:
            print("[!!!] Done")
