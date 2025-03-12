# Generated by Django 3.1.2 on 2020-11-01 14:04

import hashlib
from django.db import migrations, models


def gen_hash(apps, schema_editor):
    WebPushSubscription = apps.get_model("users", "WebPushSubscription")
    for obj in WebPushSubscription.objects.all():
        ehash = hashlib.sha1()
        ehash.update(str(obj.endpoint).encode("utf-8"))
        obj.endpoint_hash = ehash.hexdigest()
        obj.save(update_fields=["endpoint_hash"])


class Migration(migrations.Migration):
    dependencies = [
        ("users", "0035_auto_20190722_0224"),
    ]

    operations = [
        migrations.AddField(
            model_name="webpushsubscription",
            name="endpoint_hash",
            field=models.CharField(default=0, max_length=60),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name="webpushsubscription",
            name="endpoint",
            field=models.TextField(unique=False),
        ),
        migrations.RunPython(gen_hash, reverse_code=migrations.RunPython.noop),
    ]
