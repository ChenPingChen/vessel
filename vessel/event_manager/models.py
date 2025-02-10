from django.db import models
import uuid


class Vessel(models.Model):
    """船舶基本資訊"""
    detection_vessel_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    mmsi = models.CharField(max_length=9, null=True, blank=True)
    vessel_name = models.CharField(max_length=100, null=True, blank=True)
    vessel_type = models.CharField(max_length=50, null=True, blank=True)
    vessel_length = models.CharField(max_length=50, null=True, blank=True)
    vessel_width = models.CharField(max_length=50, null=True, blank=True)
    vessel_height = models.CharField(max_length=50, null=True, blank=True)
    first_seen = models.DateTimeField()
    last_seen = models.DateTimeField()
    image_url = models.CharField(max_length=255, null=True, blank=True)
    vessel_features = models.JSONField(null=True, blank=True)

    class Meta:
        db_table = 'vessel'

class VesselEvent(models.Model):
    event_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    vessel = models.ForeignKey(Vessel, on_delete=models.CASCADE)
    video_path = models.CharField(max_length=255, null=True, blank=True)
    start_time = models.DateTimeField()
    end_time = models.DateTimeField(null=True, blank=True)
    has_ais = models.BooleanField(default=False)
    duration_metadata = models.JSONField(default=dict)
    # {
    #     'tracking_points': [
    #         {
    #             'timestamp': '2024-01-20T10:00:00',
    #             'longitude': 120.xxxx,
    #             'latitude': 22.xxxx,
    #             'channel_type': 'interior_channel',
    #             'camera_positions': {
    #                 'camera1': {'longitude': 120.xxx, 'latitude': 22.xxx},
    #                 'camera2': {'longitude': 120.xxx, 'latitude': 22.xxx}
    #             }
    #         }
    #     ]
    # }
    ais_metadata = models.JSONField(null=True, blank=True)

    class Meta:
        db_table = 'vessel_event'