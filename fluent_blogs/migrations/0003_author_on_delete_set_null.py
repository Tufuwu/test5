# Generated by Django 1.11.9 on 2018-01-30 19:25

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("fluent_blogs", "0002_intro_allow_null"),
    ]

    operations = [
        migrations.AlterField(
            model_name="entry",
            name="author",
            field=models.ForeignKey(
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                to=settings.AUTH_USER_MODEL,
                verbose_name="author",
            ),
        ),
    ]
