# import the necessary packages
import uuid
import os
import sys
import urllib2
import json
import requests

class TempImage:
	def __init__(self, basePath="./", ext=".jpg"):
		# construct the file path
		self.path = "{base_path}/{rand}{ext}".format(base_path=basePath,
			rand=str(uuid.uuid4()), ext=ext)

	def cleanup(self):
		
                # remove the file
		os.remove(self.path)


// whatup 