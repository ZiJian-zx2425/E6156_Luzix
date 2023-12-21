import { ComponentFixture, TestBed } from '@angular/core/testing';

import { SearchByPatientComponent } from './search-by-patient.component';

describe('SearchByPatientComponent', () => {
  let component: SearchByPatientComponent;
  let fixture: ComponentFixture<SearchByPatientComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [SearchByPatientComponent]
    });
    fixture = TestBed.createComponent(SearchByPatientComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
