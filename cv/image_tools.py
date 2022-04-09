from queue import Empty
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import re
from scipy.ndimage import gaussian_filter
from IPython.core.display import display as display_nbk

class ImageTools():
    def __init__(self):
        self.dpi = 15
        self.data = None
        self.data_array = None
        self.in_path = ""
        self.out_path = ""
        self.selected_file_list = ""
        self.last_state = ""
        self.cmap = "gray"
        self.pipelines = {}
            
    def open(self, image_file):
        self.data = Image.open(f"{self.in_path}/{image_file}")
        self.data_array = np.array(self.data)
        self.original_filename = image_file
        self.base_filename = self.get_base_filename_from_path(image_file)
        self.original_ext = image_file.split(".")[1]
    
    def save_data_array(self):
        self.data_array = np.array(self.data)
     
    def set_in_path(self, path):
        self.in_path = path
    
    def set_out_path(self, path):
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
            print(f"The new path [{path}] is created!")
        self.out_path = path
        
    def revert(self):
        self.data = self.last_state
        self.save_data_array()
    
    def get_dpi(self):
        dpi_out = (self.dpi, self.dpi)
        try:
            dpi_out = self.data.info["dpi"]
        except:
            print("dpi not accesible")
        return dpi_out
    
    def create_plot(self):
        height, width, depth = self.data_array.shape
        # What size does the figure need to be in inches to fit the image?
        dpi = self.get_dpi()
        
        figsize = width // float(dpi[0]), height // float(dpi[1])
        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        # Hide spines, ticks, etc.
        #ax.axis('off')
        # Display the image.
        ax.imshow(self.data_array, cmap=self.cmap)
    
    def show(self, show_axis="on"):
        # Create a figure of the right size with one axes that takes up the full figure
        fig = plt.figure(figsize=self.get_figsize())
        ax = fig.add_axes([0, 0, 1, 1])
        # Hide spines, ticks, etc.
        ax.axis(show_axis)
        # Display the image.
        ax.imshow(self.data_array, cmap=self.cmap)
        plt.show()
    
    def show_without_axis(self):
        self.show(show_axis="off")
       
    def get_figsize(self):
        if len(self.data_array.shape) == 3:
            try:
                height, width, _ = self.data_array.shape
            except:
                print("problem with expected shape")
        else:
            try: 
                print(self.data_array.shape)
                height, width = self.data_array.shape
            except:
                raise Exception("problem with expected shape in get_figsize")
        dpi = self.get_dpi()
        figsize = width // float(dpi[0]), height // float(dpi[1])
        return figsize
             
    @classmethod
    def get_data_figsize(cls, data, data_array=None, dpi_default = 25):
        ''' This function calculate real figsize
            To reduce processing time, pass data_array and data
        '''
        if data_array is None:
            data_array = np.array(data)
        height, width, _ = data_array.shape
        try:
            dpi = data.info["dpi"]
        except:
            dpi = (dpi_default, dpi_default)
        figsize = width // float(dpi[0]), height // float(dpi[1])
        return figsize
    
    
    
    @classmethod    
    def show_data(cls, data, cmap="gray"):
        data_array = np.array(data)
        fig = plt.figure(figsize=ImageTools.get_data_figsize(data, data_array))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(data_array, cmap=cmap)
        plt.show()
    
    def display(self):
        plt.figure(figsize=self.get_figsize())
        display_nbk(self.data)
        
        
    def get_selected_files(self):
        return self.selected_file_list
    

    def show_red(self):
        try:
            plt.figure(figsize=self.get_figsize())
            ax = plt.imshow(self.data_array[:,:,0], cmap=self.cmap)
            return ax
        except Exception as e:
            print(e)
            print("Revise el rango de datos")
            
    def show_green(self):
        try:
            plt.figure(figsize=self.get_figsize())
            ax = plt.imshow(self.data_array[:,:,1], cmap=self.cmap)
            return ax
        except Exception as e:
            print(e)
            print("Revise el rango de datos")
                
    def show_blue(self):
        try:
            plt.figure(figsize=self.get_figsize())
            ax = plt.imshow(self.data_array[:,:,2], cmap=self.cmap)
            return ax
        except Exception as e:
            print(e)
            print("Revise el rango de datos")
    
    def show_all_channels(self):
        print("red")
        self.show_red()
        plt.show()
        print("green")
        self.show_green()
        plt.show()
        print("blue")
        self.show_blue()
        plt.show()
    
    def save_last_state(self):
        self.last_state = self.data
        #self.data_array = np.array(self.data)
        
    def convert_L(self):
        print(self)
        print(f"in convert_L, [{self.data}]")
        self.save_last_state()
        self.data = self.data.convert("L")
    
    def get_size(self):
        size = np.array(self.data).size
        return size

    def get_shape(self):
        shape = np.array(self.data).shape
        return shape
    
    def get_base_filename_from_path(self, path):
        file = os.path.split(path)[1]
        base_filename, ext = os.path.splitext(file)
        return base_filename
    
    def select_all_files_from_in_path(self):
        file_list = os.listdir(self.in_path)
        self.selected_file_list = file_list
    
    def get_all_filenames_from_in_path(self):
        file_list = os.listdir(self.in_path)
        return file_list
    
    def get_filenames_from_in_path_with_extension(self, extension):
        file_list = [f for f in os.listdir(self.in_path) if f.endswith(f".{extension}")]
        return file_list
    
    #def select_path_filenames_from_in_path_with_extension(self, extension):
    #    file_list = [os.path.join(self.in_path,f) for f in os.listdir(self.in_path) if f.endswith(f".{extension}")]
    #    self.selected_file_list = file_list
        
    def select_filenames_from_in_path_with_extension(self, extension):
        file_list = [f for f in os.listdir(self.in_path) if f.endswith(f".{extension}")]
        self.selected_file_list = file_list
    
    def select_filenames_from_in_path_with_pattern(self, pattern):
        file_list = [f for f in os.listdir(self.in_path) if re.match(pattern,f)]
        self.selected_file_list = file_list
        
    def get_filenames_from_in_path_with_pattern(self, pattern):
        file_list = [f for f in os.listdir(self.in_path) if re.match(pattern,f)]
        return file_list
    
    def save_file(self, extension_output, sufix=''):
        try:
            self.data.convert('RGB').save(f"{self.out_path}/{self.base_filename}{sufix}.{extension_output}")
        except IOError:
            raise Exception(f"Cannot convert file to {extension_output}")
            
        # input_file
    
    #def convert_selected_list_files(self):
    #    for infile in self.file_list:
    #        outfile = f"{}/{}.jpg.format(outPath) blablabla"
    
    # arreglar   
    def save_to_jpg(self):
        self.data.convert('RGB').save(f"{self.out_path}/")
        
    # revisar
    def cut_region(self, x_ini,y_ini,x_end,y_end):
        self.save_last_state()
        box = (x_ini, y_ini, x_end, y_end)
        self.data = self.data.crop(box)
        
    def rotate(self, grade):
        self.save_last_state()
        self.data = self.data.rotate(grade)
    
    # revisar
    def flip_left_right(self):
        self.save_last_state()
        self.data = self.data.transpose(Image.FLIP_LEFT_RIGHT)
        
    def flip_top_bottom(self):
        self.save_last_state()
        self.data = self.data.transpose(Image.FLIP_TOP_BOTTOM)
        
    def transpuesta(self):
        self.save_last_state()
        self.data = self.data.transpose(Image.TRANSPOSE)
    
    def transversa(self):
        self.save_last_state()
        self.data = self.data.transpose(Image.TRANSVERSE)
        
    def rotate_90(self):
        self.save_last_state()
        self.data = self.data.transpose(Image.ROTATE_90)
        
    def rotate_180(self):
        self.save_last_state()
        self.data = self.data.transpose(Image.ROTATE_180)
    
    def rotate_270(self):
        self.save_last_state()
        self.data = self.data.transpose(Image.ROTATE_270)          
    
    
    def transpose(self):
        self.save_last_state()
        self.data.transpose()
        pass
        
    def resize(self, x,y):
        self.save_last_state()
        self.data = self.data.resize((x,y))
    
    def get_array(self):
        return np.array(self.data)
        
    def get_one_dim_array(self):
        self.data_array = np.array(self.data)
        self.data_dtype = self.data_array.dtype
        self.data_shape = self.data_array.shape
        return self.data_array
    
    def get_split_rgb(self):
        return self.data.split()
    
    
    
    @staticmethod
    def pipeline(*args):
        try:
            obj = args[0]
            for arg in args[1:]:
                if isinstance(arg, list):
                    method = arg[0]
                    vars = arg[1:]
                    method(obj,*vars)
                else:
                    arg(obj)
        except Exception as e:
           print("An exception has been captured in pipeline")
           print(e)
           raise e
                
    def create_pipeline(self,name, *args):
        print(args)
        self.pipelines[name] = args
        
        
    def apply_pipeline(self, pipeline_name):
        if self.data is None :
            raise Exception("You must load a file to apply a pipeline")
        try:
            ImageTools.pipeline(self, *self.pipelines[pipeline_name])
        except Exception as e:
           print("An exception has been captured in apply_pipeline")
           raise e
            
        
    def apply_pipeline_to_selected_files(self, pipeline_name):
        if self.selected_file_list is Empty:
            print("WARNING: Selected file list empty")
        else:
            tmp = ImageTools()
            tmp.set_in_path(self.in_path)
            tmp.set_out_path(self.out_path)
            for file in self.selected_file_list:
                print(f"applying pipeline [{pipeline_name}] to file [{file}]")
                tmp.open(file)
                tmp.pipelines = self.pipelines
                tmp.apply_pipeline(pipeline_name)
            tmp = None

    def gaussian_filter_x(self, sigma=10):
        self.save_last_state()
        self.data = self.data.convert("L")
        self.data_array = np.array(self.data)        
        im = self.data_array #self.data_array
        imx_g = np.zeros(im.shape, np.uint8)
        self.data_array= gaussian_filter(im, sigma , (1,0), imx_g)
        self.data = Image.fromarray(self.data_array, mode='L')
    
    def gaussian_filter_y(self, sigma=10):
        print("in gaussian_filter_y")
        self.save_last_state()
        self.data = self.data.convert("L")
        self.data_array = np.array(self.data)        
        im = self.data_array #self.data_array
        imy_g = np.zeros(im.shape, np.uint8)
        self.data_array= gaussian_filter(im, sigma , (0,1), imy_g)
        self.data = Image.fromarray(self.data_array, mode='L')
    
    def from_array_to_image(self):
        self.data = Image.fromarray(self.data_array)
    
    #def gaussian_filter_y(self, sigma=10):
    #    print("in gaussian_filter_y")
    #    self.data = self.data.convert("L")
    #    im = np.array(self.data)
    #    imy_g= np.zeros(im.shape)
    #    gaussian_filter(im, (sigma,sigma), (1,0), imy_g)
    #    self.data_array = imy_g
    #    #self.data=Image.fromarray(self.data_array)
    
    #def gaussian_filter_y(self, sigma=10):
    #    self.convert_L()
    #    im = self.data_array
    #    #array(Image.open('data2/01_empire.jpg').convert('L'), 'f')
        #sigma = 10 #Desviacion estandar
    #    imx_g = zeros(im.shape)
    #    gaussian_filter(im, (sigma,sigma), (0,1), imx_g)
    #    imy_g= zeros(im.shape)
    #    gaussian_filter(im, (sigma,sigma), (1,0), imy_g)
    #    self.data_array = im
        # Dibujando
#        plt.close("all")
#        plt.figure()
#        plt.suptitle("Imagen y sus gradientes a lo largo de cada eje")
#        ax = plt.subplot(1,3,1)
#        ax.axis("off")
#        ax.imshow(im, cmap = plt.get_cmap('gray'))
#        ax.set_title("im_empire")

#        ax = plt.subplot(1,3,2)
#        ax.axis("off")
#        ax.imshow(imx_g, cmap = plt.get_cmap('gray'))
#        ax.set_title("gx")

#        ax = plt.subplot(1,3,3)
#        ax.axis("off")
#        ax.imshow(imy_g , cmap = plt.get_cmap('gray'))
#        ax.set_title("gy")
#        plt.show()
#        plt.clf() 
