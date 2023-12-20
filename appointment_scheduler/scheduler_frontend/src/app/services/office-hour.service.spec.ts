import { TestBed } from '@angular/core/testing';

import { OfficeHourService } from './office-hour.service';

describe('OfficeHourService', () => {
  let service: OfficeHourService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(OfficeHourService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
