#!/usr/bin/env python
#-*- coding: utf-8 -*-

#hack
import sys,os
#sys.getdefaultencoding("utf-8")

import codecs
import re
import string

import csv
import io
import warnings
import glob
import subprocess

warnings.filterwarnings("ignore")

from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import ImageDraw
from PIL import ImageFont
from scipy.ndimage import filters

import random
import math
#random.seed()

import numpy as np
import matplotlib

import matplotlib.pyplot as plt

from gtts import gTTS

########################################
# General settings (dev-code!)
########################################

DEBUGDIR = './'
INDEXDIR = './'
STIMDIR = './'

arialfont = '/Library/Fonts/Arial Bold.ttf'
fontsize = 25
backgroundcolor = 'rgb(150,150,150)'
fontcolor = 'rgb(0,0,0)'


fontdirs = [
    '/usr/local/share/fonts/',
    ]

###############################################################################
# settings for settings
###############################################################################

plain_sets={'Local':
                {'xy_res': (1280,1024),
                'xy_mar': (110,148), 
                'y_space':1.8,
                'font':"/Library/Fonts/Courier New Bold.ttf",
                'font_s':33,
                'pupcol':5,
                'tab_s':4,
                'font_col':(0,0,0),
                'back_col':(225, 225, 225),
                'dpi':(72,72),
                'zval':3,
                'index':True}}
                
            
uil_l =  {'xy_res': (1280,1024),
            'xy_mar': (110,170), 
            'y_space':1.8,
            'font':"/Library/Fonts/Arial.ttf",
            'font_s':33,
            'pupcol':5,
            'tab_s':4,
            'font_col':(0,0,0),
            'back_col':(225, 225, 225),
            'dpi':(72,72),
            'zval':3,
            'index':True}

uil_backdrop={'centerblue':
                {'layer0_s':(1280,1024),
                'layer0_c':(int(.8*256),int(.8*256),int(.8*256)),
                'layer1_s':(1024,768),
                'layer1_c':(int(0.186*256),int(.360*256),int(.630*256)),
                'layer1_p':(128,128),
                'layer2_c':(200,200,200),
                'layer2_s':(1024,68),
                'layer2_p':(128,128)}}

#shortcuts
S = plain_sets['Local']
B = uil_backdrop['centerblue']

########################################
# Hanne experiment generation settings 
########################################

###############################################################################
# Block Tutorial
# 3 sound files
# 3 image files
###############################################################################

fn_tut_i = ["Tutorial1i" , "Tutorial2i", "Tutorial3i"]
ph_tut_i = ["Rabbit(1i)", "Cat(2i)", "Cow(3i)"]

fn_tut_s = ["Tutorial1s" , "Tutorial2s", "Tutorial3s"]
ph_tut_s = ["Rabbit", "Cat", "Cow"]

def make_tutorial():
    """
    Create Tutorial stimuli
    """
    assert len(fn_tut_i) == len(ph_tut_i)
    
    for fn_im, ph_im in zip(fn_tut_i, ph_tut_i):
        wordim = make_text_in_box(text=ph_im)
        wordim.save('./' + fn_im + '.png')

    assert len(fn_tut_s) == len(ph_tut_s)

    make_word_sounds(mytextlist=ph_tut_s,
                    myfilenamelist=fn_tut_s, 
                    language='en',
                    savepath='./',
                    )

###############################################################################
# Block Training
# 9 sound files
# 9 image files
###############################################################################

fnamelist_s = [
        '_a1s',
        '_a2s',
        '_a3s',
        '_b1s',
        '_b2s',
        '_b3s',
        '_c1s',
        '_c2s',
        '_c3s'
        ]

fnamelist_i = [
        '_a1i',
        '_a2i',
        '_a3i',
        '_b1i',
        '_b2i',
        '_b3i',
        '_c1i',
        '_c2i',
        '_c3i'
        ]

textlist = [
        'ball',
        'key',
        'car',
        'chair',
        'mug',
        'phone',
        'shoe',
        'house',
        'boat'
        ]

#https://www.lexilogos.com/keyboard/pinyin_conversion.htm
pinyinlist1 = [
        'mā', 
        'má',
        'mà',
        'yī',
        'yí',
        'yì',
        'nē',
        'né',
        'nè'
        ]

#shame, but no
pinyinlist2 = [
        'ma1', 
        'ma2',
        'ma4',
        'yi1',
        'yi2',
        'yi4',
        'ne1',
        'ne2',
        'ne4'
        ]

#nope
tonelist = [1, 2, 4, 1, 2, 4, 1, 2, 4]      

def make_training():
    """
    Create Training stimuli
    """
    assert len(textlist) == len(fnamelist_i) == len(fnamelist_s)
    
    for fn_im, ph_im, py_im in zip(fnamelist_i, textlist, pinyinlist1):
        wordim = make_text_in_box(text=py_im, box=(200,200), save=False)
        wordim.save('./' + fn_im + '.png')

    make_word_sounds(mytextlist=pinyinlist2,
                    myfilenamelist=fnamelist_s, 
                    language='zh-cn',
                    savepath='./',
                    )

###############################################################################
# Block Test Series A (HAPPY)
# 9 sound files
# 9 image files
###############################################################################

test_neut = ['ma1n','ma2n','ma4n','yi1n','yi2n','yi4n','ne1n','ne2n','ne4n']
test_tone = ['1', '2', '4', '1', '2', '4', '1', '2', '4']

def make_happy_videos(): 
    for n, tone in zip(test_neut, test_tone):
        soundobj = make_word_sound(text=n, fname=n, save=True)
        name = n
        wav = n + '.wav'
        neut = f"""Name: {name}
Sound: {wav}
SoundTone: {tone}
Type: 
     |||||||||||
     |||||||||||
     |||||||||||
     NEUTRAL-ISH 

""".format(name=n, wav=n+'.wav', tone=tone)
        print (neut)
        with open('text.txt', 'w') as writer:
            writer.writelines(neut)
        writer.close()
        output = n + '.mp4'
        subprocess.call(f"""ffmpeg -f lavfi -i "color=color=black, \
                drawtext=enable='gte(t,0)':fontfile=Monaco.ttf:\
                fontcolor=white:textfile=text.txt\
                :reload=1:y=50:\
                x=(W/tw)*n" -i {wav} -c:a aac -t 3 {output}""".format(wav=wav, 
                    output=output), shell=True)

###############################################################################
# Block Test Series A (HAPPY)
# 9 sound files
# 9 image files
###############################################################################

test_happ = ['ma1h','ma2h','ma4h','yi1h','yi2h','yi4h','ne1h','ne2h','ne4h']
test_tone = ['1', '2', '4', '1', '2', '4', '1', '2', '4']

def make_neutral_videos():
    for n, tone in zip(test_happ, test_tone):
        soundobj = make_word_sound(text=n, fname=n, save=True)
        name = n
        wav = n + '.wav'
        happ = f"""Name: {name}
Sound: {wav}
SoundTone: {tone}
Type:
Type: 
     )))))))))))
     )))))))))))
     )))))))))))
     HAPPY-ISH 

""".format(name=n, wav=wav, tone=tone)
        print (happ)
        with open('text.txt', 'w') as writer:
            writer.writelines(happ)
        writer.close()
        output = n + '.mp4'
        subprocess.call(f"""ffmpeg -f lavfi -i "color=color=black, \
                drawtext=enable='gte(t,0)':fontfile=Monaco.ttf:\
                fontcolor=white:textfile=text.txt\
                :reload=1:y=50:\
                x=(W/tw)*n" -i {wav} -c:a aac -t 3 {output}""".format(wav=wav, 
                    output=output), shell=True)

#####################
# to create elements
#####################

def make_text(text=u'Bwöp!', 
        background='rgb(225, 225, 225)', 
        fontcolor='rgb(0, 0, 0)',
        fontpath='/Library/Fonts/Courier New Bold.ttf', 
        font_s=33, 
        save=False,
        ):
    """
    Make a text element as large as needed according to the font and its point size. 
    Returns PIL image instance.
    
    - text : string of text [str]
    - background : background color text [str]
    - fontcolor : font color [str]
    - fontpath : path to font [str]
    - font_s : font size (point size, so integer only) [int]
    - save : for debugging: [bool]
    """
    dum = Image.new('RGB',(100,100), background) # crash test dum 
    im = ImageDraw.Draw(dum)
    myfont = ImageFont.truetype(fontpath, font_s, encoding='unic')
    osz = im.textsize(text, font=myfont)
    image = Image.new('RGB',(osz[0], osz[1]), background)
    canvas = ImageDraw.Draw(image)
    canvas.text((0,0),text,fontcolor,font=myfont)
    if save:
        image.save('./text_element.png', 
            dpi=(300,300), 
            quality=100, 
            optimize=True)

    return image

def make_text_in_box(text=u'Bwöp!', 
        background='rgb(225, 225, 225)', 
        fontcolor='rgb(0, 0, 0)',
        fontpath='/Library/Fonts/Courier New Bold.ttf', 
        font_s=33, 
        save=False,
        box=(200, 200)
        ):
    """
    Make a text element as large as needed according to the font and its point size. 
    Returns PIL image instance.
    
    - text : string of text [str]
    - background : background color text [str]
    - fontcolor : font color [str]
    - fontpath : path to font [str]
    - font_s : font size (point size, so integer only) [int]
    - save : for debugging: [bool]
    """
    dum = Image.new('RGB',(100,100), background) # crash test dum 
    im = ImageDraw.Draw(dum)
    myfont = ImageFont.truetype(fontpath, font_s, encoding='unic')
    osz = im.textsize(text, font=myfont)
    image = Image.new('RGB',(box[0], box[1]), background)
    canvas = ImageDraw.Draw(image)
    canvas.text((0,0),text,fontcolor,font=myfont)
    if save:
        image.save('./text_element.png', 
            dpi=(300,300), 
            quality=100, 
            optimize=True)

    return image    

def make_dot(outer=(100, 100),
                    inner=(9, 9),
                    colorback='rgba(225, 225, 225, 0)',
                    colorout='rgb(0, 0, 0)', 
                    colorin='rgb(255, 255, 255)',
                    save=True
                    ):
    """
    Makes a calibration 'dot' element.
    """
    image = Image.new('RGB', outer, colorback)
    canvas = ImageDraw.Draw(image)
    canvas.ellipse((0,0, outer[0]-1, outer[1]-1), fill = colorout)
    canvas.ellipse(((outer[0]-1)//2 - inner[0], 
        (outer[1]-1)//2 -inner[1],(outer[0]-1)//2 + inner[0], 
        (outer[1]-1)//2 + inner[1]), fill=colorin)    
    if save:
        image.save('./dot_element.png', 
            dpi=(300,300), 
            quality=100, 
            optimize=True)
    return image

def make_gabor(im=None,
                xoff=0, 
                yoff=0, 
                ori=0, 
                lumi=127, 
                contrast=1, 
                sfreq=0.05, 
                std=7
                ):
    """
    Draw Gabor patch. 

    If im==None, a test image is made and saved
    
    - im : image in which to put the Gabor (PIL image instance)
    - xoff : topleft of the gabor (float)
    - yoff : topright of the gabor (float)
    - ori : orientation (in radians), (float)
    - lumi : luminance (value between 0-127)
    - contrast : contrast (0 .. 1)
    - sfreq : spatial frequency
    - std : standard deviation
    
    Adapted from <http://www.cogsci.nl>
    """
    size = 5 # gabor pixel dimension size is std * size
    imwasnone = False
    if im == None:
        imwasnone = True
        im = Image.new("L", (std * size, std * size), 127)
    px = im.load()  
    for rx in range(int(std * size)):
        for ry in range(int(std * size)):               
            dx = rx - 0.5 * size * std
            dy = ry - 0.5 * size * std      
            t = math.atan2(dy, dx) + ori
            r = math.sqrt(dx ** 2 + dy ** 2)            
            x = r * math.cos(t)
            y = r * math.sin(t)     
            i = lumi * (1 + contrast * math.cos(2 * math.pi * x * sfreq) * \
                math.exp(-0.5*(x/std)**2 - 0.5*(y/std)**2) )
            px[rx + xoff, ry + yoff] = int(i) + px[rx + xoff, ry + yoff] - 127
    if imwasnone:
        im.save('./gabor_test.png', dpi=(300,300), quality=100, optimize=True)
    return im
            
def make_tile(char=u'w', 
            fontsize=80, 
            saveloc='./', 
            font=u'/Library/Fonts/Arial.ttf'):
    """
    Create a single letter
    """
    dumim = Image.new('RGB', (40, 40), 'rgb(255,255,255)')
    im = ImageDraw.Draw(dumim)
    myfont = ImageFont.truetype(font, fontsize, encoding='unic')
    w, h = im.textsize(char, font=myfont)
    # now the real image
    tile = Image.new('RGB', (w, h), 'rgb(255,255,255)')
    tile_im = ImageDraw.Draw(tile)
    # draw the character on the canvas
    tile_im.text((0,0),char, 'rgb(0,0,0)',font=myfont)
    if saveloc:
        tile.save(saveloc + 'tile_' + char + '.png', 
            dpi=(72,72), quality=100, optimize=True)
    return tile

def find_edges(image):
    """
    Crop
    """
    image = image.convert("I")
    ima = np.asarray(image)
    vertical = np.sum(ima, axis=1)
    horizontal = np.sum(ima, axis=0)
    #print ('v', vertical)
    #print ('h', horizontal)
    # the h,v diff is not really as good as i thought, lets binarize
    boo = ima == 255
    h = []
    v = []
    # we loop over boolean rows
    hc = 0
    for row in boo:
        print (row)
        if not np.alltrue(row):
            h.append(hc)
        hc += 1
    vc = 0
    for col in boo.T:
        print (col)
        if not np.alltrue(col):
            v.append(vc)
        vc += 1
    # tight is the cropped version of the array
    tight = np.array(ima[h[0]:h[-1]+1,v[0]:v[-1]+1],copy=True)
    # test to see the cropped version
    return ima, boo, h, v, tight

###############################################################################
# helpers
###############################################################################

def resize_percent(im,percent):
    """
    Resize an image in percentage.
    """
    w,h = im.size
    return im.resize((int(percent * w /100), 
        int(percent * h/100)), Image.BICUBIC)

def resize_pix(im,pixels):
    """
    Resize an image in pixels.
    """
    (wx,wy) = im.size
    rx=1.0*wx/pixels
    ry=1.0*wy/pixels
    if rx>ry:
        rr=rx
    else:
        rr=ry
    return im.resize((int(wx//rr), int(wy//rr)))

def get_resize(got,want):
    """
    Get the precentage by which (an image) should be resized.
    """
    afactor = float(float(got)/float(want))
    factor = float(1./afactor)
    return factor * 100
    
def read_unicode(fname):
    """
    Read a unicode file and return all lines
    """
    #f = codecs.open(infile, encoding='utf-8')
    f = open(fname, encoding="utf-8")
    return f.readlines()

def make_circle(xres=(0,1280),
                yres=(0,1024),
                n=10,
                radius=350,
                inspect=True):
    """
    Make a grid of 'n' points to form a circle in a cartesian plane.
    Originating from the center, using radius as pixel distance between points. 

    - xres: tuple with minimum and maximum pixels in x dimension
    - yres: tuple with minimum and maximum pixels in y dimension
    - n: number of points
    - radius: the radius of the circle 
    
    xvec and yvec --as arrays.
    """
    centralx = 0.5 * xres[1] - xres[0]
    centraly = 0.5 * yres[1] - yres[0]
    k = np.arange(n)
    xyplaces = (radius * np.cos(float(2 * np.pi) * k/n+1), 
                radius * np.sin(float(2 * np.pi) * k/n+1))
    xvec = np.asarray(xyplaces[0] + centralx)
    yvec = np.asarray(xyplaces[1] + centraly)
    if inspect:
        plt.plot(xvec, yvec, 'k.')
        plt.axis([0,1280,0,1024])
        plt.grid('on')
        plt.savefig('./circle_coords.png')
    return xvec, yvec

def make_grid(imsize=(1280,1024),
            xmar=60,
            ymar=60,
            elx=30,
            ely=30,
            xels=8,
            yels=8,
            jitter=0,
            xy=None,
            inspect=True):
    """
    Generate a grid of evenly spaced elements. 
    With optional jitter (percentage).
    
    - imsize: tuple with min and max pixel values for the image [tuple]
    - xmar: x-margin for elements (pixels) [int]
    - ymar: y-margin for elements (pixels) [int]
    - elx: size (pixels) of one element in x dimension [int]
    - ely: size (pixels) of one element in y dimension [int]
    - xels: number of elements in x [int]
    - yels: number of elements in y [int]
    - jitter: amount of jitter (percentage) [int/float]
    - xy: no idea, best leave it None 
    - inspect: plot the genrated coordinates with matplotlib for debugging [bool]
    """
    if xy != None:
        return xy[0],xy[1]
    x_left = imsize[0] - 2 * xmar
    y_left = imsize[1] - 2 * ymar
    use_xmarge, use_ymarge = False, False
    most_right_pos = imsize[0] - xmar - elx
    most_left_pos = xmar
    most_down_pos = imsize[1] - ymar - ely
    most_up_pos = ymar
    space_x = (most_right_pos - most_left_pos)/xels
    if .5 * space_x > xmar:
        use_xmarge=True
    space_y = (most_down_pos - most_up_pos)/yels
    if .5*space_y>ymar:
        use_ymarge=True
    xpp = list(np.linspace(most_left_pos, most_right_pos,xels))
    xpp = yels * xpp 
    xpp=np.array(xpp).reshape((yels,xels)).transpose()
    for i in range(xels):
        for j in range(yels):
            if i==0 or i+1==xels:# keep things inside the canvas
                if use_xmarge:
                    xpp[i,j] = xpp[i,j] + random.uniform(-1,1) * xmar * float(jitter)/100 #but of course, only if necessary
                else:
                    xpp[i,j] = xpp[i,j] + random.uniform(-1,1) * 0.5 *(space_x) * float(jitter)/100
            else:
                xpp[i,j] = xpp[i,j] + random.uniform(-1,1) * 0.5 * (space_x) * float(jitter)/100 #half ifpossible
    xpp = np.round(xpp)
    xpp = np.array(xpp,int)
    ypp = list(np.linspace(most_up_pos, most_down_pos, yels)) #the same for the y posisionts
    ypp = xels*ypp
    ypp = np.array(ypp).reshape((xels,yels))
    for i in range(xels):
        for j in range(yels):
            if j==0 or j+1==yels:
                if use_ymarge:
                    ypp[i,j] = ypp[i,j] + random.uniform(-1,1) * ymar * float(jitter)/100
                else:
                    ypp[i,j]=ypp[i,j]+ random.uniform(-1,1) * 0.5 * (space_y) * float(jitter)/100
            else:
                ypp[i,j] = ypp[i,j] + random.uniform(-1,1) * 0.5 * (space_y) * float(jitter)/100
    ypp = np.round(ypp)
    ypp = np.array(ypp,int)
    if inspect:
        plt.clf()
        plt.plot(xpp, ypp, 'k.')
        plt.axis([0,imsize[0],0,imsize[1]])
        plt.grid('on')
        plt.savefig(u'./rectangle_coords.png')
    return xpp,ypp

def make_uil_blue(testloc='./uilskuiken.png'):
    """
    Create a basic background.
    """
    # initialize dummy canvas
    base = Image.new('RGB', (B['layer0_s'][0],B['layer0_s'][1]), B['layer0_c'])
    onbase = ImageDraw.Draw(base)
    base.save(testloc,dpi=(96,96), quality=100, optimize=True)
    
def make_uil_background(testloc='./uil.png'):
    """
    Create a basic background
    """
    # initialize dummy canvas
    base = Image.new('RGB', (B['layer0_s'][0],B['layer0_s'][1]), B['layer0_c'])
    basedraw = ImageDraw.Draw(base)
    lay1 = Image.new('RGB',(B['layer1_s'][0],B['layer1_s'][1]), B['layer1_c'])
    lay1draw = ImageDraw.Draw(lay1) 
    lay2 = Image.new('RGB',(B['layer2_s'][0],B['layer2_s'][1]), B['layer2_c'])
    lay2draw = ImageDraw.Draw(lay2)
    #resize the graphic to fit in layer2 (y-based)
    logu = Image.open('./dot_element.png');
    eye_x, eye_y = logu.size
    eyefactor = get_resize(got=eye_y, want=B['layer2_s'][1])
    smaller_eye = resize_percent(logu, eyefactor)
    # resize the text grapic
    logotext = Image.open('./gabor_test.png')
    text_x,text_y = logotext.size
    textfactor = get_resize(got=text_y, want=B['layer2_s'][1])
    smaller_text = resize_percent(logotext, textfactor)
    # paste down
    base.paste(lay1,B['layer1_p'])
    base.paste(lay2,B['layer2_p'])
    base.paste(smaller_eye,B['layer2_p'])
    # add the x-space teken by the logo for the next position to paste
    eye_xs,eye_ys = smaller_eye.size
    adjusted = (B['layer2_p'][0] + eye_xs, B['layer2_p'][1])
    base.paste(smaller_text, adjusted)
    base.save(testloc,dpi=(96,96), quality=100, optimize=True)

#speech
def make_word_sounds(mytextlist,
                    myfilenamelist, 
                    language='en',
                    savepath='./'
                    ):
    """
    Create a bunch of (single word) mp3's
    """
    assert len(mytextlist) == len(myfilenamelist)

    for mytext, myfname in zip(mytextlist, myfilenamelist):
        myobj = gTTS(text=mytext, lang=language, slow=False)
        myobj.save(savepath + myfname.replace(' ','_') + ".wav")
        print ("saved: " + mytext + " under " + myfname + ".wav")

def make_word_sound(text,
                    fname, 
                    language='en',
                    savepath='./',
                    save=False
                    ):
    """
    Create a bunch of (single word) mp3's
    """
    myobj = gTTS(text=text, lang=language, slow=False)
    if save:
        myobj.save(savepath + fname.replace(' ','_') + ".wav")
        print ("saved: " + text + " under " + fname + ".wav")
    return myobj

if __name__ == '__main__':
    make_tutorial()
    make_training()
    make_neutral_videos()
    make_happy_videos()
print("__Hello?, placeholder stimulants baked...?")


