import { TestBed } from '@angular/core/testing';

import { PatientsRecordsService } from './patients-records.service';

describe('PatientsRecordsService', () => {
  let service: PatientsRecordsService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PatientsRecordsService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
