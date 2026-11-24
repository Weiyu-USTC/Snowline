"""Public forecast points shown in the dashboard and widget."""

SITES = {
    'alta': dict(id='alta', name='Alta', location='Utah', target_label='Alta–Collins manual fresh snowfall', timezone='America/Denver', latitude=40.57607, longitude=-111.63889, elevation_m=2945, target_note='Manual station target, not a resort-wide average.'),
    'vail': dict(id='vail', name='Vail', location='Colorado', target_label='Vail CAVMM snow sensor proxy', timezone='America/Denver', latitude=39.6176, longitude=-106.38, elevation_m=3140, target_note='Reported Sno24 sensor proxy; fresh-snow equivalence and reset protocol are not verified.'),
    'mansfield': dict(id='mansfield', name='Stowe', location='Vermont · Stowe area', target_label='NOAA Mount Mansfield station', timezone='America/New_York', latitude=44.5247, longitude=-72.8153, elevation_m=1204, target_note='NOAA station snowfall; not a Stowe resort-wide total. Historical reporting times are uncertain.'),
    'jay': dict(id='jay', name='Jay Peak', location='Vermont', target_label='NOAA Jay Peak station · 561 m', timezone='America/New_York', latitude=44.9381, longitude=-72.5022, elevation_m=560.8, target_note='Lower-elevation NOAA station target, not Jay Peak summit snowfall. Historical reporting times are uncertain.'),
    'mammoth': dict(id='mammoth', name='Mammoth Mountain', location='California', target_label='Mammoth Main Lodge', timezone='America/Los_Angeles', latitude=37.6508, longitude=-119.0367, elevation_m=2713, target_note='Forecasts are for the Main Lodge area. The resort reports snowfall at Main Lodge.'),
}
