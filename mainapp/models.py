from django.db import models

# Create your models here.
import pandas as pd

class File(models.Model):
    # def __init__(self,file):
    #     self.file = models.FileField(upload_to="csv_files")
    #     self.p_df = pd.read_csv(self.file)
    file = models.FileField(upload_to="csv_files")

    def __str__(self):
        return str(self.file)