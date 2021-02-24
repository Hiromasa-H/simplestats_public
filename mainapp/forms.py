from django import forms

class UploadFileForm(forms.Form):
    uploaded_file = forms.FileField(label="ファイルを選択してください")
