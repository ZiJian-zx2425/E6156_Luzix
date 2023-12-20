import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PatientsRecordsComponent } from './patients-records.component';

describe('PatientsRecordsComponent', () => {
  let component: PatientsRecordsComponent;
  let fixture: ComponentFixture<PatientsRecordsComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [PatientsRecordsComponent]
    });
    fixture = TestBed.createComponent(PatientsRecordsComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
