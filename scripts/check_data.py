#!/usr/bin/env python3
"""Validate real dataset manifest, spot arrays, label counts, patient counts."""
import argparse
from reliability_st.data import load_manifest,check_features

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);args=p.parse_args()
    table=load_manifest(args.manifest)
    print('image/gene dimensions:',check_features(table,args.manifest))
    print('cohort sections:',table.groupby('cohort').size().to_dict())
    print('unique patients:',table.groupby('cohort').patient_id.nunique().to_dict())
    print('grade labels:',table.groupby(['cohort','grade_label']).size().to_dict())
