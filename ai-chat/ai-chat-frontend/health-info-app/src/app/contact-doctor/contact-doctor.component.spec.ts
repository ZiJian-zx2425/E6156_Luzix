import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ContactDoctorComponent } from './contact-doctor.component';

describe('ContactDoctorComponent', () => {
  let component: ContactDoctorComponent;
  let fixture: ComponentFixture<ContactDoctorComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [ContactDoctorComponent]
    });
    fixture = TestBed.createComponent(ContactDoctorComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
