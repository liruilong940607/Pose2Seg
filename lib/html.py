import dominate
from dominate.tags import *
import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import argparse
import random
import cv2


class SimpleHtml():
    def __init__(self, html_file='./index.html', refresh=0):
        self.html_file = html_file
        self.doc = dominate.document(title='simple_html')
        if refresh > 0:
            with self.doc.head:
                meta(http_equiv="reflesh", content=str(refresh))
        
    def newline(self):
        self.table = table(border=1, style="table-layout: fixed;")
        self.doc.add(self.table)
        self.tr = tr()
        self.table.add(self.tr)
        self._save()
        
    def add_image(self, im, txt, height=400, isbgr=True):
        _td = td(style="word-wrap: break-word;", halign="center", valign="top")
        with _td:
            with p():
                imgstr = self._im2str(im, isbgr)
                img(style="height:%dpx" % height, 
                    src="data:image/jpg;base64,%s" % imgstr)
                br()
                p(txt)
        
        self.tr.add(_td)
        self._save()
        
    def _im2str(self, im, isbgr):
        if len(im.shape)==3:
            if isbgr:
                pil_image = Image.fromarray(im[:,:,::-1])
            else:
                pil_image = Image.fromarray(im)
        else:
            pil_image = Image.fromarray(im)
        buff = BytesIO()
        pil_image.save(buff, format="JPEG")
        imgstr = base64.b64encode(buff.getvalue()).decode("utf-8")
        return imgstr
    
    def _save(self):
        with open(self.html_file, 'wt') as f:
            f.write(self.doc.render())

if __name__ == '__main__':
    html = SimpleHtml()
    html.newline()
    
    image = np.zeros((50, 60, 3), dtype=np.uint8) + 128
    html.add_image(image, 'test')
    image = np.zeros((50, 60), dtype=np.uint8) + 128
    html.add_image(image, 'test')
    
    html.newline()
    image = np.zeros((50, 60), dtype=np.uint8) + 128
    html.add_image(image, 'test')
