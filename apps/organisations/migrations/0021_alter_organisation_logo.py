# Generated by Django 3.2.13 on 2022-06-15 14:18

import adhocracy4.images.fields
from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("a4_candy_organisations", "0020_termsofuse_make_user_and_org_not_editable"),
    ]

    operations = [
        migrations.AlterField(
            model_name="organisation",
            name="logo",
            field=adhocracy4.images.fields.ConfiguredImageField(
                "logo",
                blank=True,
                help_text="The Logo representing your organisation. The image must be square and it should be min. 200 pixels wide and 200 pixels tall and max. 800 pixels wide and 800 pixels tall. Allowed file formats are png, jpeg, gif. The file size should be max. 5 MB.",
                upload_to="organisations/logos",
                verbose_name="Logo",
            ),
        ),
    ]
